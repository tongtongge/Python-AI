import customtkinter as ctk
import requests
import json
import threading
import queue
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 全局配置 Ollama
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen1.5:1.8b"

# 本地模型路径
model_path = "../textpro/qwen1.5-1.8b-chat"

# 全局窗口配置
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class ChatWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        # 窗口基础设置
        self.title("本地Qwen1.5流式对话工具")
        self.geometry("800x600")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 模型变量（全局复用，只加载一次）
        self.model = None
        self.tokenizer = None
        # 流式输出消息队列，用于主线程更新UI
        self.stream_queue = queue.Queue()
        self.is_generating = False

        # 1. 对话展示框
        self.chat_text = ctk.CTkTextbox(self, wrap="word")
        self.chat_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.chat_text.configure(state="disabled")

        # 2. 输入框
        self.input_box = ctk.CTkEntry(self, width=600, placeholder_text="输入你的问题...")
        self.input_box.grid(row=2, column=0, padx=10, pady=(0,10), sticky="ew")
        self.input_box.bind("<Return>", self.send_msg)

        # 3. 发送按钮
        self.send_btn = ctk.CTkButton(self, text="发送", command=self.send_msg)
        self.send_btn.grid(row=2, column=1, padx=(0,10), pady=(0,10))

        # 加载模型（单独方法，只运行一次）
        self.load_model()
        # 启动主线程UI刷新循环，处理流式输出
        self.update_stream_ui()

    # ========== 独立模型加载方法 ==========
    def load_model(self):
        """单独加载本地模型与分词器，仅初始化时执行一次"""
        try:
            self.append_text("系统：正在加载本地模型，请稍候...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                # device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.append_text("系统：模型加载完成，可以开始对话！\n")
        except Exception as e:
            self.append_text(f"系统：模型加载失败！错误信息：{str(e)}\n")

    # ========== UI辅助方法 ==========
    def append_text(self, text):
        """追加静态文本"""
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", text + "\n")
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def stream_append(self, text):
        """纯追加流式字符（不换行）"""
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", text)
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def update_stream_ui(self):
        """主线程定时读取队列，渲染流式输出（解决tkinter线程安全问题）"""
        try:
            while not self.stream_queue.empty():
                token = self.stream_queue.get_nowait()
                self.stream_append(token)
        except queue.Empty:
            pass
        # 循环刷新UI
        self.after(30, self.update_stream_ui)

    # ========== 本地模型 流式推理 ==========
    def stream_local_model(self, prompt):
        if self.model is None or self.tokenizer is None:
            self.stream_queue.put("模型未加载成功！\n")
            self.is_generating = False
            self.send_btn.configure(state="normal")
            return

        try:
            messages = [
                {"role": "system", "content": "你是一个聊天机器人，与用户交流"},
                {"role": "user", "content": prompt}
            ]
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(
                [text_input],
                return_tensors="pt",
                truncation=True
            ).to(self.model.device)

            # 流式逐token生成
            with torch.no_grad():
                generated_ids = inputs.input_ids
                for _ in range(512):
                    outputs = self.model.generate(
                        generated_ids,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    new_token_ids = outputs[:, generated_ids.shape[1]:]
                    if new_token_ids[0][0] == self.tokenizer.eos_token_id:
                        break
                    # 解码单个token，推送到UI队列
                    token_str = self.tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
                    self.stream_queue.put(token_str)
                    generated_ids = outputs
        except Exception as e:
            self.stream_queue.put(f"\n推理异常：{str(e)}")
        finally:
            self.stream_queue.put("\n")
            self.is_generating = False
            self.send_btn.configure(state="normal")

    # ========== Ollama接口方法（保留原版） ==========
    def loadOllamaAPI(self, prompt):
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "num_ctx": 32768
        }
        res = requests.post(OLLAMA_URL, json=data, timeout=120)
        resp_data = res.json()
        answer = resp_data["response"]
        return answer

    # ========== 消息发送主逻辑 ==========
    def send_msg(self, event=None):
        user_text = self.input_box.get().strip()
        if not user_text or self.is_generating:
            return

        # 显示用户消息
        self.append_text(f"你：{user_text}")
        self.append_text("AI：")
        self.input_box.delete(0, "end")

        # 锁定按钮，防止重复发送
        self.is_generating = True
        self.send_btn.configure(state="disabled")

        # 新开线程执行流式推理，不阻塞UI
        t = threading.Thread(target=self.stream_local_model, args=(user_text,))
        t.daemon = True
        t.start()

if __name__ == "__main__":
    app = ChatWindow()
    app.mainloop()