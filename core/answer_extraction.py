"""
Answer extraction and response validation utilities.

This module provides robust numerical answer extraction from model responses
and validation methods to detect incomplete or problematic responses.
"""

import re
from typing import Optional


class AnswerExtractor:
    """Robust answer extraction logic for mathematical reasoning problems."""
    
    @staticmethod
    def _to_float_from_fraction(s: str) -> Optional[float]:
        """
        Convert string to float, handling fractions, mixed numbers, and various formats.
        
        Args:
            s: String containing a number, fraction, or mixed number
            
        Returns:
            Float value or None if conversion fails
        """
        if not s:
            return None
            
        # Remove currency symbols, commas, and normalize
        s = s.strip().replace('$', '').replace('£', '').replace('€', '').replace('\\$', '')
        s = s.replace(',', '').replace('\\!', '').replace('\\,', '')
        
        # Handle LaTeX formatting like \boxed{\$2400}
        s = re.sub(r'\\[a-z]+\{', '', s)  # Remove \command{ patterns
        s = s.replace('}', '').replace('{', '')
        s = s.strip()
        
        # Mixed number like "1 2/3" or "-1 2/3"
        m = re.match(r'^([+-]?\d+)\s+(\d+)/(\d+)$', s)
        if m:
            whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return whole + (num/den if whole >= 0 else -num/den)
        
        # Simple fraction "2/3"
        m = re.match(r'^([+-]?\d+)/(\d+)$', s)
        if m:
            den = int(m.group(2))
            if den == 0:
                return None
            return float(int(m.group(1)) / den)
        
        # Plain float/int including scientific notation
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """
        Extract numerical answer with priority order:
        1. \\boxed{...} or boxed{...} (highest priority - LaTeX format)
        2. "Final Answer" phrases  
        3. #### format (GSM8K standard)
        4. Last well-formed number after equals sign
        5. Last number in text
        
        Args:
            text: Response text from the language model
            
        Returns:
            Extracted numerical value or None if extraction fails
        """
        if not text:
            return None
        text = str(text).strip()

        # 1) Look for \boxed{...} or boxed{...} anywhere - highest priority
        boxed_patterns = [
            r'\\?boxed\{([^}]+)\}',  # Captures everything inside boxed{}
        ]
        
        for pattern in boxed_patterns:
            boxed_matches = re.findall(pattern, text, re.IGNORECASE)
            if boxed_matches:
                for match in reversed(boxed_matches):
                    match = match.strip()
                    match = re.sub(r'[^\d.,\-+/\s]', '', match)
                    f = AnswerExtractor._to_float_from_fraction(match)
                    if f is not None:
                        return f

        # 2) Look for "Final Answer:", "The answer is", etc. (case insensitive)
        final_patterns = [
            r'Final\s+Answer\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)',
            r'The\s+answer\s+is\s*\$?\s*([\d,]+(?:\.\d+)?)',
            r'Answer\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)',
        ]
        
        for pattern in final_patterns:
            final_matches = re.findall(pattern, text, re.IGNORECASE)
            if final_matches:
                candidate = final_matches[-1].strip()
                f = AnswerExtractor._to_float_from_fraction(candidate)
                if f is not None:
                    return f

        # 3) Look for #### format (GSM8K standard)
        gsm_pattern = r'####\s*\$?\s*([\d,]+(?:\.\d+)?)'
        gsm_matches = re.findall(gsm_pattern, text)
        if gsm_matches:
            candidate = gsm_matches[-1].strip()
            f = AnswerExtractor._to_float_from_fraction(candidate)
            if f is not None:
                return f

        # 4) Look for equals patterns (=) followed by a number
        last_portion = text[-800:] if len(text) > 800 else text
        equals_patterns = [
            r'=\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:\n|$|\.)',
            r'=\s*\$?\s*([\d,]+(?:\.\d+)?)',
        ]
        
        for pattern in equals_patterns:
            equals_matches = re.findall(pattern, last_portion)
            if equals_matches:
                candidate = equals_matches[-1].strip()
                f = AnswerExtractor._to_float_from_fraction(candidate)
                if f is not None:
                    return f

        # 5) Last resort: find the last well-formed number in the text
        last_portion = text[-500:] if len(text) > 500 else text
        number_pattern = r'\$?\s*([\d,]+(?:\.\d+)?)'
        all_numbers = re.findall(number_pattern, last_portion)
        
        if all_numbers:
            for candidate in reversed(all_numbers[-5:]):
                f = AnswerExtractor._to_float_from_fraction(candidate)
                if f is not None and f > 0:
                    return f

        all_numbers = re.findall(number_pattern, text)
        if all_numbers:
            for candidate in reversed(all_numbers[-10:]):
                f = AnswerExtractor._to_float_from_fraction(candidate)
                if f is not None:
                    return f

        return None

    @staticmethod
    def is_response_incomplete(text: str) -> bool:
        """
        Check if the response seems incomplete or cut off.
        
        Args:
            text: Response text from the language model
            
        Returns:
            True if response appears incomplete
        """
        if not text:
            return True
            
        text = text.strip()
        
        # Response is too short
        if len(text) < 50:
            return True
        
        # Check for cut-off mid-sentence or mid-calculation
        incomplete_indicators = [
            text.endswith('='),
            text.endswith('+'),
            text.endswith('-'),
            text.endswith('*'),
            text.endswith('/'),
            text.endswith('('),
            text.endswith(','),
        ]
        
        return any(incomplete_indicators)

    @staticmethod
    def is_response_formula_only(text: str) -> bool:
        """
        Check if response contains only formulas without final numerical answer.
        
        Args:
            text: Response text from the language model
            
        Returns:
            True if response has formulas but no clear final answer
        """
        if not text:
            return True
        
        # Check if response is very short
        if len(text) < 100:
            return True
            
        # Check if it has mathematical expressions
        has_math = bool(re.search(r'[+\-*/=]', text))
        
        if not has_math:
            return False
            
        # Check if there's a clear final answer indication
        has_final_answer = bool(re.search(
            r'(final answer|the answer is|answer:|####|\\boxed)', 
            text, 
            re.IGNORECASE
        ))
        
        # If has math but no final answer indication, likely formula-only
        return has_math and not has_final_answer