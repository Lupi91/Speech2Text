import threading
import os
from time import perf_counter

import customtkinter as ctk
#from customtkinter import StringVar
import tkinter as tk
from CTkListbox import *
from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece as spm

import srt
import datetime
import time

ct_model_path = "models/NLLB/nllb-200-distilled-600M-int8"
sp_model_path = "models/NLLB/flores200_sacrebleu_tokenizer_spm.model"


def translate2(source_sents, sp, translator, tgt_lang, beam_size):
    try:
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
        #return translations
        return translations[0]
    
    except:
        return "ERROR"


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
        self.grid_rowconfigure(9, weight=1)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # ROW0: load files button
        self.audiobutton = ctk.CTkButton(self, command=self.select_audio, height=50, text="Select File/s", font=ctk.CTkFont(family="Verdana", size=14,weight="bold"))
        self.audiobutton.grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="nsew")
        # initalize list for loaded files
        self.audio = []
        
        # ROW1: listbox for loaded files
        self.input_files = CTkListbox(self, font=ctk.CTkFont(size=12)) 
        self.input_files.grid(row=1, columnspan=4, padx=20, pady=5, sticky="nsew")
        
        # ROW2: task
        self.rb_task_var = ctk.StringVar()
        self.rb3 = ctk.CTkRadioButton(self, text="transcribe", variable=self.rb_task_var, value="transcribe", command=self.radiobutton_event)
        self.rb4 = ctk.CTkRadioButton(self, text="translate", variable=self.rb_task_var, value="translate", command=self.radiobutton_event)
        self.rb3.grid(row=2, column=0, columnspan=1, padx=20, pady=10, sticky="nsew")
        self.rb4.grid(row=2, column=1, columnspan=1, padx=20, pady=10, sticky="nsew")
        self.rb3.select()
        # clear list box / remove item buttons
        self.listbox_clear = ctk.CTkButton(self, command=self.listbox_clear, height=25, text="Clear List", font=ctk.CTkFont(family="Verdana", size=12))
        self.listbox_clear.grid(row=2, column=2, padx=20, pady=10, sticky="new")
        self.remove_item = ctk.CTkButton(self, command=self.remove_item, height=25, text="Clear Selection", font=ctk.CTkFont(family="Verdana", size=12))
        self.remove_item.grid(row=2, column=3, padx=20, pady=10, sticky="new")

        
        # ROW3 - COL 0/1 (whisper settings)
        self.label1 = ctk.CTkLabel(self, text="model size", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label1.grid(row=3, column=0, padx=20, pady=10, sticky="nw")

        self.modelsize = ctk.CTkComboBox(self, values=["small", "medium", "large"], state="readonly", variable=ctk.StringVar(value="medium"))
        self.modelsize.grid(row=3, column=1, padx=20, pady=10, sticky="nsew")

        self.label2 = ctk.CTkLabel(self, text="beam_size", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label2.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        beam = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.beamsize = ctk.CTkComboBox(self, values=beam, state="readonly", variable=ctk.StringVar(value=5))
        self.beamsize.grid(row=4, column=1, padx=20, pady=10, sticky="nsew")

        self.label3 = ctk.CTkLabel(self, text="device", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label3.grid(row=5, column=0, padx=20, pady=10, sticky="nw")
        self.device = ctk.CTkComboBox(self, values=["auto", "cpu", "cuda"], state="readonly", variable=ctk.StringVar(value="auto"))
        self.device.grid(row=5, column=1, padx=20, pady=10, sticky="nsew")

        self.label4 = ctk.CTkLabel(self, text="VAD filter", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label4.grid(row=6, column=0, padx=20, pady=10, sticky="nw")
        self.vad_filter = ctk.CTkComboBox(self, values=["True", "False"], state="readonly", variable=ctk.StringVar(value="False"))
        self.vad_filter.grid(row=6, column=1, padx=20, pady=10, sticky="nsew")

        # ROW 3 - COL 2/3 (task/language)
        self.label5 = ctk.CTkLabel(self, text="Target Language", width=140, font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        self.label5.grid(row=3, column=2, padx=20, pady=10, sticky="nw")
        self.tb1 = ctk.CTkEntry(self)
        self.tb1.grid(row=4, column=2, padx=20, pady=10, sticky="nw")
        
        self.listbox = CTkListbox(self, font=ctk.CTkFont(size=10)) 
        self.listbox.grid(row=3, column=3, rowspan=5, padx=20, pady=10, sticky="nsew")
        for i, item in enumerate(self.language_list):
            self.listbox.insert(i, item)
        
        # auto update listbox based on the default init_lang.txt
        self.tb1.bind('<KeyRelease>', self.checkkey)
        self.tb1.insert("insert", self.language_init)
        self.checkkey('<KeyRelease>')
        
        # disable the translate widgets
        self.radiobutton_event()


        # output options
        #self.label6 = ctk.CTkLabel(self, text="output options", font=ctk.CTkFont(weight="bold"), anchor="w", justify="left")
        #self.label6.grid(row=7, column=0, padx=20, pady=10, sticky="nw")
        #self.cb_txt = ctk.StringVar(value="on")
        #self.cb1 = ctk.CTkCheckBox(self, text="txt", variable=self.cb_txt, onvalue="on", offvalue="off")
        #self.cb1.grid(row=7, column=1, padx=20, pady=10, sticky="nsew")
        self.cb_csv = ctk.StringVar(value="on")
        self.cb2 = ctk.CTkCheckBox(self, text="csv", variable=self.cb_csv, onvalue="on", offvalue="off")
        self.cb2.grid(row=7, column=0, padx=20, pady=10, sticky="nsew")
        self.cb_srt = ctk.StringVar(value="off")
        self.cb3 = ctk.CTkCheckBox(self, text="srt", variable=self.cb_srt, onvalue="on", offvalue="off")
        self.cb3.grid(row=7, column=1, padx=20, pady=10, sticky="nsew")
        
        # button RUN
        self.transcribebutton = ctk.CTkButton(self, command=self.transcribe_button, height=50, text="Run", font=ctk.CTkFont(family="Verdana", size=14,weight="bold"))
        self.transcribebutton.grid(row=8, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        
        
        # console / logging widget
        self.console = ctk.CTkTextbox(self)
        self.console.grid(row=9, column=0, columnspan=4, padx=20, pady=10, sticky="nsew")

    
    
    def listbox_clear(self):
        # clear listbox (loaded files)
        self.input_files.delete("all")
        del self.audio[:]
    
    def remove_item(self):
        # remove selected file
        if self.input_files.curselection() is not None:
            self.audio.remove(self.input_files.get())
            self.input_files.delete(self.input_files.curselection())
            
    def checkkey(self, event):
        # filter the listbox (languages) based on the user input / auto select
        def update(data):
            self.listbox.delete("all")
            for i, item in enumerate(data):
                self.listbox.insert(i, item)
            try:
                self.listbox.activate(0)
            except:
                pass    
        
        value = self.tb1.get()
        if not value:
            data = self.language_list
        else:
            data = []
            for item in self.language_list:
                if value.lower() in item.lower():
                    data.append(item)
        update(data)

    def radiobutton_event(self):
        # disable or enable required widgets
        if self.rb_task_var.get() == "transcribe":
            self.label5.grid_forget()
            self.tb1.grid_forget()
            self.listbox.grid_forget()
        else:
            self.label5.grid(row=3, column=2, padx=20, pady=10, sticky="nw")
            self.tb1.grid(row=4, column=2, padx=20, pady=10, sticky="nw")
            self.listbox.grid(row=3, column=3, rowspan=5, padx=20, pady=10, sticky="nsew")
            self.checkkey('<KeyRelease>')

    def write(self,*message, end = "\n", sep = " "):
        # console logger
        text = ""
        for item in message:
            text += "{}".format(item)
            text += sep
        text += end
        self.console.insert("insert", text)
        self.console.see(tk.END) # auto scrol down

    
    def select_audio(self):
        # old single file load method
        #self.audio = tk.filedialog.askopenfilename(title="Select File")
        #self.audiofilename = os.path.basename(self.audio)
        #if self.audiofilename != "":
            #self.audiobutton.configure(text=self.audiofilename)
            #self.write("File:", self.audio)
        
        # new multiple file load method using a listbox
        files = tk.filedialog.askopenfilename(title="Select File/s", multiple=True)
        # start index = actual number of items in listbox
        # avoid overwriting, allow user to load files multiple times
        existing_items = self.input_files.size()
        for idx, f in enumerate(files):
            self.input_files.insert((idx + existing_items), f)
            self.audio.append(f)
            

    def transcribe_button(self):
        threading.Thread(target=self.transcribe).start()

    def transcribe(self):
        self.transcribebutton.configure(state="disabled")
        self.console.delete('1.0', tk.END)
        if self.input_files.size() == 0:
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

        v_filter = True if (self.vad_filter.get() == "True") else False
        model_dir = "models/whisper_" + self.modelsize.get()
        self.write(f"Loading whisper model [{model_dir}] ... ")
        try:
            model = WhisperModel(model_dir, device=self.device.get())
        except Exception as e:
            self.write("ERROR: Loading model failed!")
            self.write(e)
            self.transcribebutton.configure(state="normal")
            return


        self.write("Running Speech Recognition...")
        t1 = perf_counter()
        for idx, audio_file in enumerate(self.audio):
            try:
                self.write("_____________________________________________________________")
                segments, info = model.transcribe(audio_file, beam_size=int(self.beamsize.get()), task=self.rb_task_var.get(), vad_filter=v_filter)
                self.write(f"Processing: {os.path.basename(audio_file)}")
            except Exception as e:
                self.write("(%s/%s) ERROR processing file: %s" % (idx+1, len(self.audio), os.path.basename(audio_file)))
                self.write(str(e))
                if idx == len(self.audio) - 1:
                    self.transcribebutton.configure(state="normal")
                    return
                else:
                    continue
            
            RESULT = []
            RESULT_CSV = []
            TRANSLATION = []
            SUBTITLES = []
            
            self.write("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            for segment in segments:
                self.write("(%s/%s) [%.2fs -> %.2fs] %s" % (idx+1, len(self.audio), segment.start, segment.end, segment.text))
                RESULT.append(segment.text[1:])
                
                if self.rb_task_var.get() == "translate":
                    translated_str = translate2(segment.text[1:], sp, translator, tgt_lang, int(self.beamsize.get()))
                    TRANSLATION.append(translated_str)
                    #self.write("[translation] %s" % (translated_str))
                
                if self.cb_csv.get() == "on":
                    #start_time = datetime.timedelta(milliseconds=segment.start * 1000)
                    #end_time = datetime.timedelta(milliseconds=segment.end * 1000)
                    start_time = time.strftime('%H:%M:%S', time.gmtime(segment.start))
                    end_time = time.strftime('%H:%M:%S', time.gmtime(segment.end))

                    if self.rb_task_var.get() == "translate":
                        #RESULT_CSV.append("%.2fs;%.2fs;%s;%s" % (segment.start, segment.end, segment.text[1:], translated_str))
                        RESULT_CSV.append("%s;%s;%s;%s" % (start_time, end_time, segment.text[1:].replace(";",","), translated_str.replace(";",",")))               
                    else:    
                        #RESULT_CSV.append("%.2fs;%.2fs;%s" % (segment.start, segment.end, segment.text[1:]))
                        RESULT_CSV.append("%s;%s;%s" % (start_time, end_time, segment.text[1:].replace(";",",")))

                # srt generation, credits: https://github.com/IOriens/whisper-video    
                if self.cb_srt.get() == "on":
                    i = 0
                    start_time = datetime.timedelta(milliseconds=segment.start * 1000)
                    end_time = datetime.timedelta(milliseconds=segment.end * 1000)
                    if self.rb_task_var.get() == "translate":
                        #text = translated_str[0]
                        text = translated_str
                    else:
                        text = segment.text.strip()
                    
                    if text:
                        # Create a subtitle object for the segment
                        subtitle = srt.Subtitle(index=i + 1, start=start_time, end=end_time, content=text)
                        SUBTITLES.append(subtitle)
                        i += 1

            
            # save output
            if any(RESULT):
                txt_output = os.path.join(os.path.dirname(audio_file), (os.path.basename(audio_file) + "_transcript.txt"))
                #txt_output = os.path.splitext(audio_file)[0] + ".txt"
                with open(txt_output, "w", encoding='utf-8') as f:
                    for ln in RESULT:
                        f.write("%s\n" % ln)

                    if self.rb_task_var.get() == "translate":
                        f.write("\n### TRANSLATION\n")
                        for ln in TRANSLATION:
                            f.write("%s\n" % ln)

                self.write("\nSaved transcript to: \n" + txt_output)
            
                
                if self.cb_csv.get() == "on":
                    csv_output = os.path.splitext(audio_file)[0] + ".csv"
                    with open(csv_output, "w", encoding="utf-8-sig") as f:
                        for ln in RESULT_CSV:
                            f.write("%s\n" % ln)
                    self.write(csv_output)

                
                if self.cb_srt.get() == "on":
                    srt_output = os.path.splitext(audio_file)[0] + ".srt"
                    with open(srt_output, "w", encoding="utf-8") as f:
                        f.write(srt.compose(SUBTITLES))
                        
                    self.write(srt_output)

            else:
                self.write("NO SPEECH DETECTED!")
        
        self.write("_____________________________________________________________")
        t2 = perf_counter()
        self.write("DONE! - Elapsed time:", round((t2-t1), 2), "sec")
        self.transcribebutton.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()