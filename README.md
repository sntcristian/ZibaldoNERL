# ZibaldoNERL

## Named Entity Recognition and Linking on Giacomo Leopardi's Zibaldone


## Install Requirements

Due to dependency issues, the bi-encoder requires a different huggingface version than LLMs. For this reason, we suggest to create two different conda environments.

### Create BELA (bi-encoder) environment

```
conda create -n bela39 -y python=3.9 && conda activate bela39
pip install -r requirements_bela.txt
```

### Create LLM environment
```
conda create -n llm -y python=3.10 && conda activate llm
pip install -r requirements_llms.txt
```

## Citation

```
@inproceedings{santini_named_2024,
	address = {Amsterdam, Netherlands},
	title = {Named {Entity} {Recognition} in {Historical} {Italian}: {The} {Case} of {Giacomo} {Leopardi}’s {Zibaldone}},
	volume = {3967},
	url = {https://ceur-ws.org/Vol-3967/X-TAIL-2024_paper_1.pdf},
	urldate = {2025-05-29},
	booktitle = {Posters and {Demos}, {Workshops}, and {Tutorials} of {EKAW} 2024},
	publisher = {CEUR},
	author = {Santini, Cristian and Melosi, Laura and Frontoni, Emanuele},
	year = {2024},
}
```



## References

* Santini, C. (2024). Zibaldoned: Silver annotations for Entity Disambiguation from Digitalzibaldone (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13971759


* Stoyanova, S. & Johnston, B. (Eds.), *Giacomo Leopardi's Zibaldone di pensieri: a digital research platform*. https://digitalzibaldone.net/

* Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2024). GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. In K. Duh, H. Gomez, & S. Bethard (A c. Di), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) (pp. 5364–5376). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-long.300

* Plekhanov, M., Kassner, N., Popat, K., Martin, L., Merello, S., Kozlovskii, B. M., Dreyer, F. A., & Cancedda, N. (2023). Multilingual End to End Entity Linking. arXiv preprint arXiv:2306.08896. 

* Mistral AI. (2025). Mistral-Small-24B-Instruct-2501 [Large language model]. Hugging Face. https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501

