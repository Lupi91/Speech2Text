import argparse
from pathlib import Path
import sys
import os
from time import perf_counter  

from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece as spm

ct_model_path = "models/NLLB/nllb-200-distilled-600M-int8"
sp_model_path = "models/NLLB/flores200_sacrebleu_tokenizer_spm.model"

def translate2(source_sents, sp, translator):
    source_sentences = [source_sents]

    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn" # default output is German
    # List of language codes, see: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

    beam_size = 5

    source_sentences = [sent.strip() for sent in source_sentences]
    target_prefix = [[tgt_lang]] * len(source_sentences)

    # Subword the source sentences
    source_sents_subworded = sp.encode_as_pieces(source_sentences)
    source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]

    # Translate the source sentences
    translations_subworded = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=2024, beam_size=beam_size, target_prefix=target_prefix)
    translations_subworded = [translation.hypotheses[0] for translation in translations_subworded]
    for translation in translations_subworded:
      if tgt_lang in translation:
        translation.remove(tgt_lang)

    # Desubword the target sentences
    translations = sp.decode(translations_subworded)
    return translations



def main(model, input, output, task):
    model_dir = "models/whisper_medium"
    if model == "large":
        model_dir = "models/whisper_large"
    print("Loading whisper model...")
    try:
        model = WhisperModel(model_dir) # base, medium
    except Exception as e:
        print(e)
        sys.exit(1)
    
    if task == "translate":
        try:
            print("Loading sentencepiece model...")
            sp = spm.SentencePieceProcessor()
            sp.load(sp_model_path)
        except Exception as e:
            print(e)
            sys.exit(1)
        
        try:
            print("Loading translator model...")
            translator = ctranslate2.Translator(ct_model_path, device="cuda")
            print("CUDA enabled!")
        except:
            try:
                translator = ctranslate2.Translator(ct_model_path, device="cpu")
                print("CPU enabled!")
            except Exception as e:
                print(e)
                sys.exit(1)

    print("")
    print("\nRunning Speech Recognition...")
    print("")
    t1 = perf_counter() 
    RESULT_FILES = []
    
    input_files = [x for x in input]
    for idx, f in enumerate(input_files):
        try:
            print("_____________________________________________________________")
            print(f"Processing: {f}")
            segments, info = model.transcribe(f, beam_size=5, task=task)
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        
        except Exception as e:
            print("(%s/%s) ERROR processing file: %s" % (idx+1, len(input_files), f))
            print("")


        RESULT = []
        TRANSLATION = []
        for segment in segments:
            if task == "translate":
                translated_str = translate2(segment.text[1:], sp, translator)
                print("(%s/%s) [%.2fs -> %.2fs] %s:" % (idx+1, len(input_files), segment.start, segment.end, translated_str))
                TRANSLATION.append(translated_str)
                RESULT.append(segment.text[1:])
            else:
                print("(%s/%s) [%.2fs -> %.2fs] %s" % (idx+1, len(input_files), segment.start, segment.end, segment.text))
                RESULT.append(segment.text[1:])
        
        
        if output:
            outfile = os.path.join(output, Path(f).stem + ".txt")
        else:
            outfile = os.path.join(os.path.dirname(f), Path(f).stem + ".txt")
        
        with open(outfile, 'a', encoding="utf-8") as f:
            for ln in RESULT:
                f.write("%s\n" % ln)
            if task == "translate":
                f.write("\n### TRANSLATION\n")
                for ln in TRANSLATION:
                    f.write("%s\n" % ln)

        RESULT_FILES.append(outfile)
    
    print("_____________________________________________________________")
    t2 = perf_counter()
    print("DONE!")
    for i in RESULT_FILES:
        print("Output:", i)
    print("Elapsed time:", round((t2-t1), 2))

from gooey import Gooey, GooeyParser
@Gooey(program_name="FASTER-WHISPER / CTranslate2 NLLB", required_cols=1, optional_cols=1)
def parse_args():   
    parser = GooeyParser(description="Speech Recognition / Translation") 
    parser.add_argument("-i", "--input", required=True, nargs='*', help="Select audio or video file/s", widget="MultiFileChooser")
    parser.add_argument("-o", "--output", required=False, help='output directory [default = input directory]', widget="DirChooser")
    parser.add_argument("-m", "--model", choices=["medium", "large"], default="medium", help="choose model type")
    parser.add_argument("-t", "--task", choices=["transcribe", "translate"], default="transcribe", help="choose task: transcribe or translate")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_args()
    main(**vars(opt))