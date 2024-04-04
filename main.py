import threading
import os
from time import perf_counter

import customtkinter as ctk
from customtkinter import StringVar
import tkinter as tk
from CTkListbox import *
from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece as spm
#os.environ["CT2_VERBOSE"] = "-1"    # -1 = error https://opennmt.net/CTranslate2/environment_variables.html
ct_model_path = "models/NLLB/nllb-200-distilled-600M-int8"
sp_model_path = "models/NLLB/flores200_sacrebleu_tokenizer_spm.model"


def translate2(source_sents, sp, translator, tgt_lang, beam_size):
    src_lang = "eng_Latn"
    source_sentences = [source_sents]

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



class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        with open("config/lang_list.txt", encoding="utf-8") as f:
            self.language_list = [line.rstrip() for line in f]
        with open("config/lang_code.txt", encoding="utf-8") as f:
            self.language_code = [line.rstrip() for line in f]
        with open("config/init_lang.txt", encoding="utf-8") as f:
            self.language_init = f.readline().strip('\n')

        self.geometry("800x800")
        self.title("Faster-Whisper-Translator")
        self.resizable(False,False)
        ctk.set_default_color_theme("dark-blue")
        self.grid_rowconfigure(7, weight=1)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)


        self.audiobutton = ctk.CTkButton(self, command=self.select_audio, height=50, text="Select File", font=ctk.CTkFont(family="Verdana", size=16,weight="bold"))
        self.audiobutton.grid(row=0, columnspan=4, padx=20, pady=10, sticky="nsew")
        self.audiofilename = ""

        # COL 0/1 (whisper settings)
        self.label1 = ctk.CTkLabel(self, text="model size", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label1.grid(row=2, column=0, padx=20, pady=10, sticky="nw")

        self.modelsize = ctk.CTkComboBox(self, values=["small", "medium", "large"], state="readonly", variable=ctk.StringVar(value="medium"))
        self.modelsize.grid(row=2, column=1, padx=20, pady=10, sticky="nw")

        self.label2 = ctk.CTkLabel(self, text="beam_size", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label2.grid(row=3, column=0, padx=20, pady=10, sticky="nw")
        beam = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.beamsize = ctk.CTkComboBox(self, values=beam, state="readonly", variable=ctk.StringVar(value=5))
        self.beamsize.grid(row=3, column=1, padx=20, pady=10, sticky="nw")

        self.label3 = ctk.CTkLabel(self, text="device", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label3.grid(row=4, column=0, padx=20, pady=10, sticky="nw")
        self.device = ctk.CTkComboBox(self, values=["auto", "cpu", "cuda"], state="readonly", variable=ctk.StringVar(value="auto"))
        self.device.grid(row=4, column=1, padx=20, pady=10, sticky="nw")

        self.label4 = ctk.CTkLabel(self, text="VAD filter", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label4.grid(row=5, column=0, padx=20, pady=10, sticky="nw")
        self.vad_filter = ctk.CTkComboBox(self, values=["True", "False"], state="readonly", variable=ctk.StringVar(value="False"))
        self.vad_filter.grid(row=5, column=1, padx=20, pady=10, sticky="nw")


        # COL 2/3 (task/language)
        self.rb_task_var = StringVar()
        self.rb3 = ctk.CTkRadioButton(self, text="transcribe", variable=self.rb_task_var, value="transcribe", command=self.radiobutton_event)
        self.rb4 = ctk.CTkRadioButton(self, text="translate", variable=self.rb_task_var, value="translate", command=self.radiobutton_event)
        self.rb3.grid(row=2, column=2, columnspan=1, padx=20, pady=10, sticky="nw")
        self.rb4.grid(row=2, column=3, columnspan=1, padx=20, pady=10, sticky="nw")
        self.rb3.select()

        self.label5 = ctk.CTkLabel(self, text="Target Language", width=140, font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label5.grid(row=3, column=2, padx=20, pady=10, sticky="nw")
        self.tb1 = ctk.CTkEntry(self)
        self.tb1.grid(row=4, column=2, padx=20, pady=10, sticky="nw")
        
        self.listbox = CTkListbox(self, width=140, font=ctk.CTkFont(size=10))
        self.listbox.grid(row=3, column=3, rowspan=4, padx=20, pady=10, sticky="nsew")
        for i, item in enumerate(self.language_list):
            self.listbox.insert(i, item)

        self.tb1.bind('<KeyRelease>', self.checkkey)
        self.tb1.insert("insert", self.language_init)
        self.checkkey('<KeyRelease>')


        self.transcribebutton = ctk.CTkButton(self, command=self.transcribe_button, height=50, text="Run", font=ctk.CTkFont(family="Verdana", size=16,weight="bold"))
        self.transcribebutton.grid(row=6, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        self.console = ctk.CTkTextbox(self)
        self.console.grid(row=7, column=0, columnspan=4, padx=20, pady=10, sticky="nsew")
        self.radiobutton_event()



    def update(self, data):
        self.listbox.delete("all")
        for i, item in enumerate(data):
            self.listbox.insert(i, item)
        try:
            self.listbox.activate(0)
        except: # IndexError or KeyError:
            pass

    def checkkey(self, event):
        value = self.tb1.get()
        if not value:
            data = self.language_list
        else:
            data = []
            for item in self.language_list:
                if value.lower() in item.lower():
                    data.append(item)
        self.update(data)

    def radiobutton_event(self):
        if self.rb_task_var.get() == "transcribe":
            self.label5.grid_forget()
            self.tb1.grid_forget()
            self.listbox.grid_forget()
        else:
            self.label5.grid(row=3, column=2, padx=20, pady=10, sticky="nw")
            self.tb1.grid(row=4, column=2, padx=20, pady=10, sticky="nw")
            self.listbox.grid(row=3, column=3, rowspan=4, padx=20, pady=10, sticky="nsew")
            self.checkkey('<KeyRelease>')

    def write(self,*message, end = "\n", sep = " "):
        text = ""
        for item in message:
            text += "{}".format(item)
            text += sep
        text += end
        self.console.insert("insert", text)
        self.console.see(tk.END) # auto scrol down

    def select_audio(self):
        self.audio = tk.filedialog.askopenfilename(title="Select File")
        self.audiofilename = os.path.basename(self.audio)
        if self.audiofilename != "":
            self.audiobutton.configure(text=self.audiofilename)
            self.write("File:", self.audio)

    def transcribe_button(self):
        threading.Thread(target=self.transcribe).start()

    def transcribe(self):
        self.transcribebutton.configure(state="disabled")
        self.console.delete('1.0', tk.END)
        if self.audiofilename == "":
            self.write("ERROR: No file selected!")
            self.transcribebutton.configure(state="normal")
            return

        if self.rb_task_var.get() == "translate":
            try:
                if self.listbox.curselection() is None:
                    self.write("ERROR: Target language missing!")
                    self.transcribebutton.configure(state="normal")
                    return
                lang_index = self.language_list.index(self.listbox.get())
                tgt_lang = self.language_code[lang_index]
            except:
                self.write("ERROR: Target language missing!")
                self.transcribebutton.configure(state="normal")
                return
            if tgt_lang == "":
                self.write("ERROR: Target language missing!")
                self.transcribebutton.configure(state="normal")
                return

            try:
                sp = spm.SentencePieceProcessor()
                sp.load(sp_model_path)
            except Exception as e:
                self.write("ERROR: Failed to load sentencepiece model!")
                self.write(e)
                self.transcribebutton.configure(state="normal")
                return

            try:
                self.write(f"Target Language: {tgt_lang}")
                self.write("Loading translator model... ")
                translator = ctranslate2.Translator(ct_model_path, device=self.device.get())
            except Exception as e:
                self.write(e)
                self.transcribebutton.configure(state="normal")
                return


        model_dir = "models/whisper_" + self.modelsize.get()
        self.write(f"Loading whisper model [{model_dir}] ... ")
        try:
            model = WhisperModel(model_dir, device=self.device.get())
        except Exception as e:
            self.write("ERROR: Loading model failed!")
            self.write(e)
            self.transcribebutton.configure(state="normal")
            return


        self.write("\nRunning Speech Recognition...")
        t1 = perf_counter()
        RESULT = []
        TRANSLATION = []
        try:
            v_filter = True if (self.vad_filter.get() == "True") else False
            segments, info = model.transcribe(self.audio, beam_size=int(self.beamsize.get()), task=self.rb_task_var.get(), vad_filter=v_filter)
        except Exception as e:
            self.write("ERROR")
            self.write(e)
            self.transcribebutton.configure(state="normal")
            return

        self.write("_____________________________________________________________")
        self.write(f"Processing: {self.audio}")
        self.write("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            self.write("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            RESULT.append(segment.text[1:])
            if self.rb_task_var.get() == "translate":
                translated_str = translate2(segment.text[1:], sp, translator, tgt_lang, int(self.beamsize.get()))
                TRANSLATION.append(translated_str)

        output = os.path.join(os.path.dirname(self.audio), self.audiofilename + "_transcript.txt")
        with open(output, "w", encoding='utf-8') as f:
            for ln in RESULT:
                f.write("%s\n" % ln)

            if self.rb_task_var.get() == "translate":
                f.write("\n### TRANSLATION\n")
                for ln in TRANSLATION:
                    f.write("%s\n" % ln)

            self.write("\nSaved transcript to " + output)

        self.write("_____________________________________________________________")
        t2 = perf_counter()
        self.write("DONE! - Elapsed time:", round((t2-t1), 2), "sec")
        self.transcribebutton.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()