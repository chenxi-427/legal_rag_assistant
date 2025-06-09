#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从中国政府网获取《中华人民共和国劳动法》全文并保存到 data 目录。
使用 Selenium 来应对动态加载的页面。
"""

import time
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# --- 配置 ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
URL = "http://www.gov.cn/banshi/2005-05/25/content_905.htm"
OUTPUT_FILE = DATA_DIR / "labor_law_full.txt"

def fetch_law_text_with_selenium():
    """
    通过 Selenium 驱动的浏览器从中国政府网获取劳动法全文。
    """
    print(f"正在从以下URL获取内容: {URL}")
    
    # 配置 Chrome 浏览器选项
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无头模式，不在前台显示浏览器窗口
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")

    try:
        # 自动安装和管理 ChromeDriver
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 访问页面
        driver.get(URL)
        
        # 等待页面加载，特别是动态内容
        time.sleep(5) # 简单等待，确保JS执行完毕
        
        # 获取页面源码
        page_source = driver.page_source
        
        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(page_source, 'html.parser')

        # 最终确认：内容在 <font id="Zoom"> 标签内
        content_div = soup.find('font', id='Zoom')

        if not content_div:
            print("错误：无法在页面上找到 id='Zoom' 的font标签。网页结构可能已更改。")
            # 保存HTML以便调试
            with open(DATA_DIR / "debug_page.html", 'w', encoding='utf-8') as f:
                f.write(page_source)
            print(f"已保存当前页面HTML到 {DATA_DIR / 'debug_page.html'} 以便分析。")
            return False

        # 提取所有文本内容
        raw_text = content_div.get_text(separator='\n', strip=True)
        
        # --- 文本清理 ---
        text_lines = raw_text.split('\n')
        
        cleaned_lines = []
        for line in text_lines:
            # 过滤掉页脚和分享链接等无关信息
            if "【E-mail推荐" in line or "【打印】" in line or "【关闭】" in line:
                break
            if line: # 只添加非空行
                cleaned_lines.append(line.strip())
        
        full_text = "\n".join(cleaned_lines)

        # 确保data目录存在
        DATA_DIR.mkdir(exist_ok=True, parents=True)

        # 将清理后的文本写入文件
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        print(f"成功！《中华人民共和国劳动法》全文已保存至: {OUTPUT_FILE}")
        return True

    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    fetch_law_text_with_selenium()