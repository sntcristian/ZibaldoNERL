import torch
import faiss
import hydra
from hydra.experimental import compose, initialize_config_module
from typing import List, Dict, Any, Tuple, Optional
import logging
import sqlite3
import re
from huggingface_hub import hf_hub_download
import os

logger = logging.getLogger(__name__)


class EntityDisambiguator:
    """Clean, functional entity disambiguation model using BELA."""

    def __init__(
            self,
            checkpoint_path: Optional[str] = None,
            faiss_index_path: Optional[str] = None,
            wikidata_index_path: Optional[str] = None,
            db_path: Optional[str] = None,
            embedding_dim: int = 300,
            config_name: str = "joint_el_mel",
            device: str = "cuda:0",
            # Hugging Face parameters
            hf_model_name: Optional[str] = None,
            cache_dir: Optional[str] = None
    ):
        """
        Initialize the entity disambiguation model.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            faiss_index_path: Path to the precomputed FAISS index
            wikidata_index_path: Path to the Wikidata QID index file
            db_path: Path to the SQLITE database containing entity information
            embedding_dim: Embedding dimension
            config_name: Hydra config name for the model
            device: Device to run the model on
            hf_model_name: Hugging Face model name (e.g., "sntcristian/WikiBELA")
            cache_dir: Cache directory for downloaded files
        """

        self.device = torch.device(device)
        self.embedding_dim = embedding_dim

        # Load from Hugging Face if model name provided
        if hf_model_name:
            self._load_from_hf(hf_model_name, cache_dir)
        else:
            # Use provided paths or defaults
            self.checkpoint_path = checkpoint_path or "./models/model_wiki.ckpt"
            self.faiss_index_path = faiss_index_path or "./models/faiss.index"
            self.wikidata_index_path = wikidata_index_path or "./models/index.txt"
            self.db_path = db_path or "./models/knowledge_base.sqlite"

        # Load model and components
        self._load_model(config_name)
        self._load_faiss_index()
        self._load_entity_db()

        logger.info(f"EntityDisambiguator initialized")

    def _load_from_hf(self, model_name: str, cache_dir: Optional[str]) -> None:
        """Load model files from Hugging Face Hub."""
        
        files = {
            'checkpoint_path': 'model_wiki.ckpt',
            'faiss_index_path': 'faiss.index',
            'wikidata_index_path': 'index.txt',
            'db_path': 'knowledge_base.sqlite'
        }
        
        for attr, filename in files.items():
            setattr(self, attr, hf_hub_download(
                repo_id=model_name,
                filename=filename,
                cache_dir=cache_dir
            ))

    def _load_model(self, config_name: str) -> None:
        """Load the BELA model from checkpoint."""
        logger.info("Loading BELA model...")

        with initialize_config_module("bela/conf"):
            cfg = compose(config_name=config_name)
            cfg.task.load_from_checkpoint = self.checkpoint_path
            cfg.task.embedding_dim = self.embedding_dim
            cfg.datamodule.ent_catalogue_idx_path = self.wikidata_index_path
            cfg.datamodule.train_path = None
            cfg.datamodule.val_path = None
            cfg.datamodule.test_path = None

        # Initialize components
        self.transform = hydra.utils.instantiate(cfg.task.transform)
        datamodule = hydra.utils.instantiate(cfg.datamodule, transform=self.transform)
        self.task = hydra.utils.instantiate(cfg.task, datamodule=datamodule, _recursive_=False)

        # Setup and move to device
        self.task.setup("train")
        self.task.eval()
        self.task.to(self.device)

    def _load_faiss_index(self) -> None:
        """Load the precomputed FAISS index."""
        logger.info(f"Loading FAISS index from {self.faiss_index_path}")
        self.faiss_index = faiss.read_index(self.faiss_index_path)

        # Move to GPU if available
        if self.device.type == 'cuda' and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, self.device.index, self.faiss_index)

    def _load_entity_db(self):
        logger.info(f"Loading entity database from {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        logger.info("Database loaded")

    def _encode_text(self, texts: List[str], mention_offsets: List[List[int]],
                     mention_lengths: List[List[int]]) -> torch.Tensor:
        """
        Encode text and extract mention representations.

        Args:
            texts: List of input texts
            mention_offsets: List of mention start positions for each text
            mention_lengths: List of mention lengths for each text

        Returns:
            Tensor of mention representations
        """
        # Prepare batch
        batch = {
            "texts": texts,
            "mention_offsets": mention_offsets,
            "mention_lengths": mention_lengths,
        }

        # Transform inputs
        model_inputs = self.transform(batch)
        token_ids = model_inputs["input_ids"].to(self.device)
        mention_offsets_tensor = model_inputs["mention_offsets"]
        mention_lengths_tensor = model_inputs["mention_lengths"]

        with torch.no_grad():
            # Encode text
            _, text_encodings = self.task.encoder(token_ids)
            text_encodings = self.task.project_encoder_op(text_encodings)

            # Extract mention representations
            mention_representations = self.task.span_encoder(
                text_encodings, mention_offsets_tensor, mention_lengths_tensor
            )

            # Filter out empty mentions
            valid_mentions = mention_representations[mention_lengths_tensor != 0]

        return valid_mentions

    def _search_candidates(self, mention_representations: torch.Tensor, k: int = 1) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Search for entity candidates using FAISS index.

        Args:
            mention_representations: Tensor of mention representations
            k: Number of candidates to retrieve

        Returns:
            Tuple of (scores, indices)
        """
        scores, indices = self.faiss_index.search(mention_representations.detach().cpu().numpy(), k=k)
        return torch.from_numpy(scores).to(self.device), torch.from_numpy(indices).to(self.device)

    def get_entity_info(self, lang, entity_id):
        supported_lang = ["en", "fr", "sv", "it", "de", "nl", "fi"]
        if lang not in supported_lang:
            raise Exception("lang must be one of the following iso-codes: en, de, fr, it, nl, fi, sv")
        
        table_name = f"{lang}wiki"
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT 
            t1.id,
            t1.wikidata_qid,
            t1.type_,
            t1.min_date,
            t2.label,
            t2.descr
        FROM entities t1
        LEFT JOIN {} t2 ON t1.id = t2.id
        WHERE t1.id = ?;
        '''.format(table_name), (entity_id,))
        result = cursor.fetchall()[0]
        return result

    def get_candidates_batch(self,
                texts: List[str],
                mention_offsets: List[List[int]],
                mention_lengths: List[List[int]],
                k: int = 10,
                lang: str = "en"
        ) -> List[List[Dict[str, Any]]]:
            """
            Get top-k candidates in a batch of texts.

            Args:
                texts: List of input texts
                mention_offsets: List of mention start positions for each text
                mention_lengths: List of mention lengths for each text

            Returns:
                List of predictions for each text, where each prediction contains:
                - start_pos: Start position of the mention
                - end_pos: End position of the mention
                - entity: Predicted entity ID
                - score: Confidence score
            """
            # Encode mentions
            mention_representations = self._encode_text(texts, mention_offsets, mention_lengths)

            # Search for candidates
            scores, indices = self._search_candidates(mention_representations, k=k)
            scores, indices = scores.tolist(), indices.tolist()

            # Format predictions
            predictions = []
            example_idx = 0
            for text, lengths in zip(texts, mention_lengths):
                candidates = []
                length = lengths[0]
                if length > 0:  # Valid mention
                    ex_indices = indices[example_idx]
                    ex_scores = scores[example_idx]
                    for index, score in zip(ex_indices, ex_scores):
                        candidate_info = self.get_entity_info(lang, index)
                        candidates.append({
                            "wb_id": candidate_info[1],
                            "type": candidate_info[2] if candidate_info[2] else "",
                            "min_date": candidate_info[3] if candidate_info[3] else "",
                            "label":candidate_info[4].replace("_", " ") if candidate_info[4] else "",
                            "descr":candidate_info[5] if candidate_info[5] else "",
                            "score": score
                        })
                predictions.append(candidates)
                example_idx+=1

            return predictions


# Convenience function for the original interface
def load_disambiguator(
        models_path: str = "./models",
        device: str = "cuda:0",
        embedding_dim: int = 300
):
    checkpoint_path = os.path.join(models_path, "model_wiki.ckpt")
    faiss_index_path = os.path.join(models_path, "faiss.index")
    wikidata_index_path = os.path.join(models_path, "index.txt")
    db_path = os.path.join(models_path, "knowledge_base.sqlite")

    return EntityDisambiguator(
        checkpoint_path=checkpoint_path,
        faiss_index_path=faiss_index_path,
        wikidata_index_path=wikidata_index_path,
        db_path=db_path,
        device=device,
        embedding_dim=embedding_dim
    )