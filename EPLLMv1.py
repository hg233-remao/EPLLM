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

events = queue.Queue() 

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

def save_config() -> None:
    '''保存配置'''
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'talk_dir': TALK_DIR, # talk 文件所在目录
                    'talk_file': TALK_FILE # 当前选中的 talk 文件
                },
                f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_config = load_config() # 读取配置文件
TALK_DIR = _config.get('talk_dir') or os.path.join(app_dir(), 'user', 'talk') 
'''
从elona+的存储结构上讲，talk文件只能放在user/talk目录下
保留这种选择的原因是方便以后做持久化
'''
if not os.path.isdir(TALK_DIR):
    TALK_DIR = os.path.join(app_dir(), 'user', 'talk') # 默认为user/talk
TALK_FILE = _config.get('talk_file') or ''
if TALK_FILE and not os.path.isfile(TALK_FILE):
    TALK_FILE = ''

selected_vender: pv.VendorSpec = pv.VENDORS["deepseek"]
provider: pv.ChatProvider = None

def replace_once(target_file): 
    '''使用当前选定的模型提供商和密钥对指定文件进行替换'''
    global provider
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
    
def before():
    global all_threading,isCreating
    isCreating=True
    try:
        if not TALK_FILE:
            events.put(('', '未选择要生成的 talk 文件'))
            return
        if not os.path.isfile(TALK_FILE):
            events.put((TALK_FILE, '选中的文件不存在了，请重新选择'))
            return
        t = threading.Thread(target=replace_once,
                             args=(TALK_FILE,),
                             name=os.path.basename(TALK_FILE)+'_threading')
        all_threading.append(t)
        t.start()
        t.join()
    except Exception as e:
        events.put((TALK_FILE, f'读取失败：{e}'))
    finally:
        all_threading.clear()
        isCreating=False
def count_time():
    global pause,t,t1,label21,label31, events
    if not pause and not isCreating:
        t1 -= 1
        label31.config(text=t1)
        if t1 == 0:
            be = threading.Thread(target=before,name='before_threading')
            be.start()
            t1 = t
    elif isCreating:
        label31.config(text='生成中...')
        t1 = t
    if not isCreating:
        label31.config(text=t1)

    while not events.empty():
        path, msg = events.get()
        messagebox.showwarning('提示', f'{path}\n{msg}')
    
    root.after(1000, count_time)
def topause():
    global pause,button1,avaliable
    if not avaliable:
        return
    if pause:
        pause = False
        button1.config(text='暂停倒计时')
    else:
        pause = True
        button1.config(text='开始倒计时')
def tplus():
    global t,label21
    t+=100
    label21.config(text=t)
def tplusp():
    global t,label21
    t+=1000
    label21.config(text=t)
def tminus():
    global t,label21
    if t>100:
        t-=100
    label21.config(text=t)
def tminusp():
    global t,label21
    if t>=1100:
        t-=1000
    label21.config(text=t)
def reset():
    global t1,label31
    t1 = t
    label31.config(text=t1)

def verify_apikey():
    '''验证密钥是否可用'''

    global entry, provider, avaliable
    if avaliable: # 如果已经验证过密钥并且可用，则不再重复验证
        return

    key = entry.get().strip()
    if not key:
        messagebox.showerror('错误', '还没填密钥呢TAT')
        avaliable = False
        provider = None
        return

    # 验证密钥是否可用
    try:
        provider = pv.make_provider(provider_var.get(), api_key = key, max_tokens = 32768)
    except pv.ProviderError as e:
        messagebox.showerror('错误', f'密钥验证失败：{str(e)}')
        provider = None
        avaliable = False
        return

    # 探查模型提供商
    pr = provider.probe()
    if not pr:
        messagebox.showerror('错误', f'密钥验证失败：{pr.message}')
        avaliable = False
        provider = None
        return
    if not pr.has_model(provider.model):
        messagebox.showerror('错误', f'密钥验证失败：提供商不支持模型 {provider.model}')
        avaliable = False
        provider = None
        return

    # 模型候选：接口返回的 > 目录里预设的 > 当前模型（去重且保持顺序）
    spec = pv.VENDORS[provider_var.get()]
    choices = list(dict.fromkeys(list(pr.models) or list(spec.models) or [provider.model]))
    provider_model_menu["values"] = choices
    provider_model_var.set(provider.model if provider.model in choices else choices[0])
    provider.model = provider_model_var.get()

    # 验证成功，provider已在前面被赋值
    avaliable = True

    # 验证通过后才把模型下拉菜单显示出来
    model_row.pack_forget()
    if len(choices) > 1:
        model_row.pack(pady=6, before=label2)

    tip = pr.message + (f'\n余额：{pr.balance}' if pr.balance else '')
    messagebox.showinfo('成功', f'密钥验证成功！\n{tip}')

def on_provider_model_change(_event=None):
    '''设置provider的模型为下拉菜单中选定的模型'''
    global provider
    if provider is not None:
        provider.model = provider_model_var.get()

def on_vender_change(vendor_id):
    '''切换provider的供应商'''
    global selected_vender, provider, avaliable
    selected_vender = pv.VENDORS[vendor_id]
    provider = None
    avaliable = False
    provider_menu.config(text=selected_vender.name)
    print(f"Selected vendor changed to: {selected_vender.name}")

    # 换了供应商，旧密钥和旧模型列表都作废，模型下拉菜单先藏起来
    provider_model_var.set("")
    provider_model_menu["values"] = ()
    model_row.pack_forget()

def finish_countdown_immediately():
    '''立即完成当前倒计时'''
    global t1, label31
    t1 = 1
    label31.config(text=t1)

def refresh_talk_label():
    '''显示当前维护的talk文件'''
    if TALK_FILE:
        talk_dir_label.config(text=f'talk 文件：{os.path.basename(TALK_FILE)}')
    else:
        talk_dir_label.config(text='talk 文件：未选择（点右侧「选择文件…」）')

def choose_talk_file():
    '''选择要生成的 talk 文件并保存配置'''
    global TALK_FILE, TALK_DIR
    start = os.path.dirname(TALK_FILE) if TALK_FILE else (
        TALK_DIR if os.path.isdir(TALK_DIR) else app_dir())

    # 打开文件选择对话框
    chosen = filedialog.askopenfilename(
        title='选择要生成的 talk 文件',
        initialdir=start,
        filetypes=[('talk 文件', '*.txt'), ('所有文件', '*.*')])
    
    if not chosen:
        return
    TALK_FILE = os.path.normpath(chosen)
    TALK_DIR = os.path.dirname(TALK_FILE)
    save_config()
    refresh_talk_label()
if __name__ == '__main__':
    #promptfile=open('talkai终极模板.txt',mode='r',encoding='utf-8')
    #prompt=promptfile.read()
    client=None
    isCreating=False
    t = 7200
    t1=t
    pause=True
    avaliable=False
    root = tk.Tk()
    root.title("EPLLMv1")
    root.geometry("600x720")
    selected_vender: pv.VendorSpec = pv.VENDORS["deepseek"]

    # 提供商下拉菜单
    provider_label = tk.Label(root, text="选择模型提供商：")
    provider_label.pack(pady=10)
    provider_var = tk.StringVar(value="deepseek")
    provider_menu = tk.OptionMenu(root, provider_var, *pv.VENDORS.keys(),
                                  command = on_vender_change)
    provider_menu.config(width=20)
    provider_menu.pack(pady=10)

    # 当前使用的 talk 目录
    talk_row = tk.Frame(root)
    talk_row.pack(pady=2)
    talk_dir_label = tk.Label(talk_row, text='', fg='#666666',
                              wraplength=340, justify='left')
    talk_dir_label.pack(side=tk.LEFT)
    talk_dir_button = tk.Button(talk_row, text='选择文件…', command=choose_talk_file)
    talk_dir_button.pack(side=tk.LEFT, padx=4)
    refresh_talk_label()

    # 密钥输入框和验证按钮
    label1 = tk.Label(root,text='密钥（复制：ctrl+c；粘贴：ctrl+v）：')
    label1.pack(pady=10)
    entry = tk.Entry(root,width=40)
    entry.pack(pady=10)
    verify_apikeybut = tk.Button(root,text='验证密钥',command=verify_apikey)
    verify_apikeybut.pack(pady=10)

    # 提供商模型下拉菜单
    # 这一菜单在选择并验证密钥后才会显示
    model_row = tk.Frame(root)
    provider_model_label = tk.Label(model_row, text="选择模型：")
    provider_model_label.pack(side=tk.LEFT)
    provider_model_var = tk.StringVar()
    provider_model_menu = ttk.Combobox(model_row, textvariable=provider_model_var,
                                       state="readonly", width=30)
    provider_model_menu.pack(side=tk.LEFT)
    provider_model_menu.bind("<<ComboboxSelected>>", on_provider_model_change)

    label2 = tk.Label(root,text=f'产生对话后倒计时重置为：（单位：秒）（下限100秒）')
    label2.pack(pady=10)
    label21 = tk.Label(root,text=t)
    label21.pack(pady=10)
    label3 = tk.Label(root,text=f'对话产生倒计时：（单位：秒）')
    label3.pack(pady=10)
    label31 = tk.Label(root,text=t1)
    label31.pack(pady=10)
    button1 = tk.Button(root, text="开始倒计时", command=topause)
    button1.pack(pady=10)
    button_hasten = tk.Button(root, text="立即完成当前倒计时", command=finish_countdown_immediately)
    button_hasten.pack(pady=10)
    button21 = tk.Button(root, text="增加产生对话后倒计时100秒", command=tplus)
    button21.pack(pady=10)
    button21p = tk.Button(root, text="增加产生对话后倒计时1000秒", command=tplusp)
    button21p.pack(pady=10)
    button22 = tk.Button(root, text="减少产生对话后倒计时100秒", command=tminus)
    button22.pack(pady=10)
    button22p = tk.Button(root, text="减少产生对话后倒计时1000秒", command=tminusp)
    button22p.pack(pady=10)
    button3 = tk.Button(root, text="重置对话产生倒计时", command=reset)
    button3.pack(pady=10)
    all_threading=[]
    count_time()
    root.mainloop()