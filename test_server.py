#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试HTTP服务器
"""

import http.server
import socketserver

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"服务器运行在 http://localhost:{PORT}")
    httpd.serve_forever() 