#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的本地LLM实现
"""

from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class SimpleLLM(LLM):
    """
    简单的本地LLM实现，用于演示目的
    实际使用时，可以替换为任何支持LangChain接口的LLM，如GPT-4等
    """
    
    temperature: float = 0.1
    
    @property
    def _llm_type(self) -> str:
        return "simple-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        使用简单的规则生成回答
        这里只是一个演示用的实现，实际应用中应替换为真实的LLM API调用
        """
        # 基于提示词中的内容生成简单回答
        # 从提示中提取问题和上下文
        try:
            # 尝试解析提示中的问题和上下文
            sections = prompt.split("[")
            context_section = next((s for s in sections if s.startswith("已知信息]")), "")
            question_section = next((s for s in sections if s.startswith("问题]")), "")
            
            context = context_section.replace("已知信息]\n", "").strip()
            question = question_section.replace("问题]\n", "").strip()
            
            # 如果上下文为空或者问题与上下文无关，返回通用回答
            if not context or len(context) < 10:
                return "根据提供的法律法规信息，无法找到与您问题直接相关的具体条款。建议您咨询专业的法律顾问获取更准确的信息。"
            
            # 优化的回答生成逻辑
            # 实际应用中应替换为真实的LLM
            
            # 提取用户问的是哪一条法律条款
            article_number = None
            if "第" in question and "条" in question:
                import re
                match = re.search(r'第([一二三四五六七八九十百零\d]+)条', question)
                if match:
                    article_number = match.group(0)
            
            # 从上下文中找到匹配的条款
            relevant_article = None
            if article_number:
                import re
                # 尝试从上下文中找到精确匹配的法条
                lines = context.split('\n')
                for line in lines:
                    if article_number in line:
                        relevant_article = line.strip()
                        break
            
            if relevant_article:
                answer = f"《中华人民共和国劳动法》{article_number}规定：\n\n{relevant_article}"
            else:
                # 如果找不到精确匹配，则生成更通用的回答
                # 清理上下文中的换行符，使回答更连贯
                cleaned_context = context.replace('\n\n', ' ').replace('\n', ' ')
                answer = f"根据相关法律条款，{cleaned_context[:200]}..."
                if article_number:
                    answer = f"抱歉，未能找到《劳动法》{article_number}的完整内容。但根据相关法律规定：\n\n{cleaned_context[:250]}..."
            
            return answer
            
        except Exception as e:
            # 异常处理，返回一个安全的回答
            return "抱歉，我无法处理您的问题。请确保您的问题清晰明确，并尝试重新提问。"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"temperature": self.temperature} 