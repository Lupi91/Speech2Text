**Simple GUI for Speech2Text using Faster-Whisper and optionally translation using CTranslate2 / NLLB**<br>
<br>![](/Demo.png?raw=true)

## Setup
```
pip install -r requirements.txt
```
Tested with WinPython 3.11 and CUDA V11.7


## Download models
The download_models.py will automatically download the required models from Hugging Face.
```
python download_models.py
```

## GPU Execution (CUDA)
GPU execution requires the following NVIDIA libraries to be installed: <br>
- cuBLAS and cuDNN for CUDA 11 <br>

## Default Translation (target language)
The default language is loaded from "config/init_lang.txt". Replace "German" with a language listed here: <br>
https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200 <br>
Do not touch the other txt-files! <br>
<br>![](/Translation1.png?raw=true)
<br>
<br> You can change the target language during runtime, use the entry widget to filter the listbox. <br>
<br>![](/Translation2.png?raw=true)

## References<br>
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
- NLLB 200 with CTranslate2: https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090
