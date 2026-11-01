# type: ignore
import json
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import providers as pv
from prompt import promptstr

def app_dir() -> str:
    '''打包成 exe 后取 exe 所在目录；源码运行时取 .py 所在目录'''
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(app_dir(), 'config.json') # 配置文件路径

def load_config() -> dict:
    '''读取上次的选择'''
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {} # 如果未读取到配置，返回空字典
    except Exception:
        return {}

def save_config(talk_dir: str, talk_file: str) -> None:
    '''保存配置'''
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'talk_dir': talk_dir, # talk 文件所在目录
                    'talk_file': talk_file # 当前选中的 talk 文件
                },
                f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def replace_once(provider, target_file, events):
    '''使用指定提供商对文件做一次重写；不依赖任何界面状态'''
    with open(target_file, mode='r', encoding='gbk') as f:
        content = f.read()

    try:
        r = provider.chat(promptstr, content)
    except pv.ProviderError as e:      # 网络/密钥/限流等，别让线程默默死掉
        events.put((target_file, e.user_message()))
        return

    if r.truncated:
        events.put((target_file, '输出被截断')) # 将事件放入队列中，供主线程处理
        return

    text = r.text.replace('\\n', '\n')
    text = re.sub(r'(?i)<\s*br\s*/?\s*>', '\n', text)   # 有些模型会用 <BR> 代替换行
    with open(target_file, mode='w', encoding='gbk') as f:
        f.write(text)


# ==================================================================
# 界面与调度：控件和状态全部挂在实例上，不再使用模块级全局变量
# ==================================================================

class App:
    '''主窗口。所有界面元素、运行时状态和回调都在这个类里'''

    def __init__(self, root: tk.Tk):
        self.root = root

        # 运行时状态
        self.events = queue.Queue()               # 子线程 → 主线程的事件队列
        self.provider = None                      # 当前验证通过的 provider
        self.avaliable = False                    # 密钥是否已验证（沿用原拼写）
        self.pause = True                         # 倒计时是否暂停
        self.t = 7200                             # 每轮间隔（秒）
        self.t1 = self.t                          # 当前剩余秒数
        self.is_creating = False                  # 是否正在生成
        self.all_threading = []                   # 本轮生成用到的线程

        # ---- 上次的选择 ----
        config = load_config() # 读取配置文件
        # 从elona+的存储结构上讲，talk文件只能放在user/talk目录下
        # 保留这种选择的原因是方便以后做持久化
        self.talk_dir = config.get('talk_dir') or os.path.join(app_dir(), 'user', 'talk')
        if not os.path.isdir(self.talk_dir):
            self.talk_dir = os.path.join(app_dir(), 'user', 'talk') # 默认为user/talk
        self.talk_file = config.get('talk_file') or ''
        if self.talk_file and not os.path.isfile(self.talk_file):
            self.talk_file = ''

        self.selected_vender: pv.VendorSpec = pv.VENDORS["deepseek"]

        self._build_ui()

    # 界面
    def _build_ui(self):
        '''创建全部控件'''
        self.root.title("EPLLMv1")
        self.root.geometry("600x720")

        # 提供商下拉菜单
        provider_label = tk.Label(self.root, text="选择模型提供商：")
        provider_label.pack(pady=10)
        self.provider_var = tk.StringVar(value="deepseek")
        self.provider_menu = tk.OptionMenu(self.root, self.provider_var,
                                           *pv.VENDORS.keys(),
                                           command=self.on_vender_change)
        self.provider_menu.config(width=20)
        self.provider_menu.pack(pady=10)

        # 当前使用的 talk 目录
        talk_row = tk.Frame(self.root)
        talk_row.pack(pady=2)
        self.talk_dir_label = tk.Label(talk_row, text='', fg='#666666',
                                       wraplength=340, justify='left')
        self.talk_dir_label.pack(side=tk.LEFT)
        talk_dir_button = tk.Button(talk_row, text='选择文件…',
                                    command=self.choose_talk_file)
        talk_dir_button.pack(side=tk.LEFT, padx=4)
        self.refresh_talk_label()

        # 密钥输入框和验证按钮
        label1 = tk.Label(self.root, text='密钥（复制：ctrl+c；粘贴：ctrl+v）：')
        label1.pack(pady=10)
        self.entry = tk.Entry(self.root, width=40)
        self.entry.pack(pady=10)
        verify_apikeybut = tk.Button(self.root, text='验证密钥',
                                     command=self.verify_apikey)
        verify_apikeybut.pack(pady=10)

        # 提供商模型下拉菜单
        # 这一菜单在选择并验证密钥后才会显示
        self.model_row = tk.Frame(self.root)
        provider_model_label = tk.Label(self.model_row, text="选择模型：")
        provider_model_label.pack(side=tk.LEFT)
        self.provider_model_var = tk.StringVar()
        self.provider_model_menu = ttk.Combobox(self.model_row,
                                                textvariable=self.provider_model_var,
                                                state="readonly", width=30)
        self.provider_model_menu.pack(side=tk.LEFT)
        self.provider_model_menu.bind("<<ComboboxSelected>>",
                                      self.on_provider_model_change)

        self.label2 = tk.Label(self.root, text='产生对话后倒计时重置为：（单位：秒）（下限100秒）')
        self.label2.pack(pady=10)
        self.label21 = tk.Label(self.root, text=self.t)
        self.label21.pack(pady=10)
        label3 = tk.Label(self.root, text='对话产生倒计时：（单位：秒）')
        label3.pack(pady=10)
        self.label31 = tk.Label(self.root, text=self.t1)
        self.label31.pack(pady=10)
        self.button1 = tk.Button(self.root, text="开始倒计时", command=self.topause)
        self.button1.pack(pady=10)
        button_hasten = tk.Button(self.root, text="立即完成当前倒计时",
                                  command=self.finish_countdown_immediately)
        button_hasten.pack(pady=10)
        button21 = tk.Button(self.root, text="增加产生对话后倒计时100秒", command=self.tplus)
        button21.pack(pady=10)
        button21p = tk.Button(self.root, text="增加产生对话后倒计时1000秒", command=self.tplusp)
        button21p.pack(pady=10)
        button22 = tk.Button(self.root, text="减少产生对话后倒计时100秒", command=self.tminus)
        button22.pack(pady=10)
        button22p = tk.Button(self.root, text="减少产生对话后倒计时1000秒", command=self.tminusp)
        button22p.pack(pady=10)
        button3 = tk.Button(self.root, text="重置对话产生倒计时", command=self.reset)
        button3.pack(pady=10)

    # 生成
    def before(self):
        '''本轮生成：把选中的文件交给一个线程去重写'''
        self.is_creating = True
        try:
            if not self.talk_file:
                self.events.put(('', '未选择要生成的 talk 文件'))
                return
            if not os.path.isfile(self.talk_file):
                self.events.put((self.talk_file, '选中的文件不存在了，请重新选择'))
                return
            t = threading.Thread(target=replace_once,
                                 args=(self.provider, self.talk_file, self.events),
                                 name=os.path.basename(self.talk_file)+'_threading')
            self.all_threading.append(t)
            t.start()
            t.join()
        except Exception as e:
            self.events.put((self.talk_file, f'读取失败：{e}'))
        finally:
            self.all_threading.clear()
            self.is_creating = False

    def count_time(self):
        '''每秒一次：走倒计时、触发生成、把队列里的事件弹出来'''
        if not self.pause and not self.is_creating:
            self.t1 -= 1
            self.label31.config(text=self.t1)
            if self.t1 == 0:
                be = threading.Thread(target=self.before, name='before_threading')
                be.start()
                self.t1 = self.t
        elif self.is_creating:
            self.label31.config(text='生成中...')
            self.t1 = self.t
        if not self.is_creating:
            self.label31.config(text=self.t1)

        while not self.events.empty():
            path, msg = self.events.get()
            messagebox.showwarning('提示', f'{path}\n{msg}')

        self.root.after(1000, self.count_time)

    # 倒计时相关
    def topause(self):
        '''开始 / 暂停倒计时'''
        if not self.avaliable:
            return
        if self.pause:
            self.pause = False
            self.button1.config(text='暂停倒计时')
        else:
            self.pause = True
            self.button1.config(text='开始倒计时')

    def tplus(self):
        self.t += 100
        self.label21.config(text=self.t)

    def tplusp(self):
        self.t += 1000
        self.label21.config(text=self.t)

    def tminus(self):
        if self.t > 100:
            self.t -= 100
        self.label21.config(text=self.t)

    def tminusp(self):
        if self.t >= 1100:
            self.t -= 1000
        self.label21.config(text=self.t)

    def reset(self):
        self.t1 = self.t
        self.label31.config(text=self.t1)

    def finish_countdown_immediately(self):
        '''立即完成当前倒计时'''
        self.t1 = 1
        self.label31.config(text=self.t1)

    # 密钥与供应商
    def verify_apikey(self):
        '''验证密钥是否可用'''
        if self.avaliable: # 如果已经验证过密钥并且可用，则不再重复验证
            return

        key = self.entry.get().strip()
        if not key:
            messagebox.showerror('错误', '还没填密钥呢TAT')
            self.avaliable = False
            self.provider = None
            return

        # 验证密钥是否可用
        try:
            self.provider = pv.make_provider(self.provider_var.get(),
                                             api_key=key, max_tokens=32768)
        except pv.ProviderError as e:
            messagebox.showerror('错误', f'密钥验证失败：{str(e)}')
            self.provider = None
            self.avaliable = False
            return

        # 探查模型提供商
        pr = self.provider.probe()
        if not pr:
            messagebox.showerror('错误', f'密钥验证失败：{pr.message}')
            self.avaliable = False
            self.provider = None
            return
        if not pr.has_model(self.provider.model):
            messagebox.showerror('错误',
                                 f'密钥验证失败：提供商不支持模型 {self.provider.model}')
            self.avaliable = False
            self.provider = None
            return

        # 模型候选：接口返回的 > 目录里预设的 > 当前模型（去重且保持顺序）
        spec = pv.VENDORS[self.provider_var.get()]
        choices = list(dict.fromkeys(list(pr.models) or list(spec.models)
                                     or [self.provider.model]))
        self.provider_model_menu["values"] = choices
        self.provider_model_var.set(
            self.provider.model if self.provider.model in choices else choices[0])
        self.provider.model = self.provider_model_var.get()

        # 验证成功，provider已在前面被赋值
        self.avaliable = True

        # 验证通过后才把模型下拉菜单显示出来
        self.model_row.pack_forget()
        if len(choices) > 1:
            self.model_row.pack(pady=6, before=self.label2)

        tip = pr.message + (f'\n余额：{pr.balance}' if pr.balance else '')
        messagebox.showinfo('成功', f'密钥验证成功！\n{tip}')

    def on_provider_model_change(self, _event=None):
        '''设置provider的模型为下拉菜单中选定的模型'''
        if self.provider is not None:
            self.provider.model = self.provider_model_var.get()

    def on_vender_change(self, vendor_id):
        '''切换provider的供应商'''
        self.selected_vender = pv.VENDORS[vendor_id]
        self.provider = None
        self.avaliable = False
        self.provider_menu.config(text=self.selected_vender.name)
        print(f"Selected vendor changed to: {self.selected_vender.name}")

        # 换了供应商，旧密钥和旧模型列表都作废，模型下拉菜单先藏起来
        self.provider_model_var.set("")
        self.provider_model_menu["values"] = ()
        self.model_row.pack_forget()

    # talk 文件
    def refresh_talk_label(self):
        '''显示当前维护的talk文件'''
        if self.talk_file:
            self.talk_dir_label.config(
                text=f'talk 文件：{os.path.basename(self.talk_file)}')
        else:
            self.talk_dir_label.config(
                text='talk 文件：未选择（点右侧「选择文件…」）')

    def choose_talk_file(self):
        '''选择要生成的 talk 文件并保存配置'''
        start = os.path.dirname(self.talk_file) if self.talk_file else (
            self.talk_dir if os.path.isdir(self.talk_dir) else app_dir())

        # 打开文件选择对话框
        chosen = filedialog.askopenfilename(
            title='选择要生成的 talk 文件',
            initialdir=start,
            filetypes=[('talk 文件', '*.txt'), ('所有文件', '*.*')])

        if not chosen:
            return
        self.talk_file = os.path.normpath(chosen)
        self.talk_dir = os.path.dirname(self.talk_file)
        save_config(self.talk_dir, self.talk_file)
        self.refresh_talk_label()


if __name__ == '__main__':
    #promptfile=open('talkai终极模板.txt',mode='r',encoding='utf-8')
    #prompt=promptfile.read()
    app = App(tk.Tk())
    app.count_time()
    app.root.mainloop()
