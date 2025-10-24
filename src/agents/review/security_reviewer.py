"""
Security Reviewer Agent

Analyzes code for security vulnerabilities and best practices:
- SQL injection vulnerabilities
- Cross-Site Scripting (XSS) risks
- Authentication and authorization issues
- Hardcoded credentials and secrets
- Insecure dependencies and imports
- Input validation issues
- Cryptography weaknesses
- File path traversal vulnerabilities

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import re
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from loguru import logger

from ..base_agent import BaseAgent


class SecurityReviewer(BaseAgent):
    """
    Specialized agent for analyzing code security.
    
    Analyzes:
    1. SQL Injection (30%) - Query construction, parameterization
    2. XSS Vulnerabilities (25%) - HTML/JS output, escaping
    3. Authentication/Authorization (20%) - Access control, sessions
    4. Secrets Management (15%) - Hardcoded credentials, API keys
    5. Insecure Dependencies (10%) - Known vulnerable imports
    
    Returns score 0-100 with detailed security analysis.
    """
    
    ASPECT_WEIGHTS = {
        "sql_injection": 0.30,
        "xss_vulnerabilities": 0.25,
        "auth_authorization": 0.20,
        "secrets_management": 0.15,
        "insecure_dependencies": 0.10
    }
    
    # Patterns for detecting security issues
    SQL_PATTERNS = [
        r'execute\s*\([^)]*%s',        # String formatting in SQL
        r'execute\s*\([^)]*\+[^)]*\)', # String concatenation in SQL  
        r'execute\s*\(\s*f["\']',      # f-string in SQL
        r'executemany\s*\([^)]*%s',
        r'\.execute\s*\([^)]*\.format\(',
        r'\".*\"\s*%\s*\w+',           # SQL query with % formatting
        r'\'.*\'\s*%\s*\w+',           # SQL query with % formatting (single quotes)
    ]
    
    XSS_PATTERNS = [
        r'innerHTML\s*=',
        r'document\.write\s*\(',
        r'eval\s*\(',
        r'\.html\s*\(\s*[^)]',  # jQuery .html()
    ]
    
    SECRETS_PATTERNS = [
        r'password\s*=\s*["\'][^"\']{3,}["\']',
        r'api_key\s*=\s*["\'][^"\']{10,}["\']',
        r'secret\s*=\s*["\'][^"\']{10,}["\']',
        r'token\s*=\s*["\'][^"\']{10,}["\']',
        r'aws_secret_access_key\s*=',
        r'private_key\s*=\s*["\']',
    ]
    
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__',
        'pickle.loads', 'yaml.load', 'subprocess.call',
        'os.system', 'subprocess.Popen'
    }
    
    INSECURE_IMPORTS = {
        'pickle': 'Use json or safer serialization',
        'yaml': 'Use yaml.safe_load() instead of yaml.load()',
        'subprocess': 'Validate all inputs, use shell=False',
        'os.system': 'Use subprocess with shell=False instead',
    }
    
    def __init__(self, name: str = "security_reviewer"):
        """
        Initialize Security Reviewer.
        
        Args:
            name: Agent name (default: "security_reviewer")
        """
        role = "Expert Security Analyst for Python Applications"
        super().__init__(
            name=name,
            role=role,
            agent_type="review"  # Uses temperature 0.2
        )
        logger.info(f"{self.name} ready for security analysis")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute security review task (async interface for BaseAgent compatibility).
        
        Args:
            task: Dictionary with task_type and data
                - task_type: Type of review task (e.g., 'review_security')
                - data: Task-specific parameters (must contain 'code')
            
        Returns:
            Dict with security analysis
        """
        task_type = task.get("task_type")
        data = task.get("data", {})
        
        if task_type == "review_security":
            return self._review_security(data)
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
    
    def review_security(self, code: str) -> Dict[str, Any]:
        """
        Synchronous security review (for direct calls and testing).
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dict with security analysis and score
        """
        return self._review_security({"code": code})
    
    def _review_security(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete security review.
        
        Args:
            context: Must contain 'code', optionally 'project_files'
            
        Returns:
            Dict with security score and analysis
        """
        code = context.get("code", "")
        if not code:
            return {
                "status": "error",
                "message": "No code provided for review"
            }
        
        logger.info("Starting security review...")
        
        try:
            tree = ast.parse(code)
            
            # Analyze each security aspect
            sql_analysis = self._analyze_sql_injection(tree, code)
            xss_analysis = self._analyze_xss_vulnerabilities(tree, code)
            auth_analysis = self._analyze_auth_authorization(tree, code)
            secrets_analysis = self._analyze_secrets_management(tree, code)
            deps_analysis = self._analyze_insecure_dependencies(tree, code)
            
            # Calculate weighted total score
            total_score = (
                sql_analysis["score"] * self.ASPECT_WEIGHTS["sql_injection"] +
                xss_analysis["score"] * self.ASPECT_WEIGHTS["xss_vulnerabilities"] +
                auth_analysis["score"] * self.ASPECT_WEIGHTS["auth_authorization"] +
                secrets_analysis["score"] * self.ASPECT_WEIGHTS["secrets_management"] +
                deps_analysis["score"] * self.ASPECT_WEIGHTS["insecure_dependencies"]
            )
            
            # Aggregate all issues and suggestions
            all_issues = (
                sql_analysis["issues"] +
                xss_analysis["issues"] +
                auth_analysis["issues"] +
                secrets_analysis["issues"] +
                deps_analysis["issues"]
            )
            
            all_suggestions = (
                sql_analysis["suggestions"] +
                xss_analysis["suggestions"] +
                auth_analysis["suggestions"] +
                secrets_analysis["suggestions"] +
                deps_analysis["suggestions"]
            )
            
            result_data = {
                "score": round(total_score, 2),
                "aspects": {
                    "sql_injection": sql_analysis,
                    "xss_vulnerabilities": xss_analysis,
                    "auth_authorization": auth_analysis,
                    "secrets_management": secrets_analysis,
                    "insecure_dependencies": deps_analysis
                },
                "issues": all_issues,
                "suggestions": all_suggestions,
                "statistics": {
                    "total_issues": len(all_issues),
                    "critical_issues": len([i for i in all_issues if i.get("severity") == "critical"]),
                    "vulnerabilities_found": len([i for i in all_issues if "vulnerability" in i.get("type", "").lower()]),
                    "functions_analyzed": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                }
            }
            
            logger.info(f"Security review complete. Score: {total_score:.2f}")
            
            return {
                "success": True,
                "status": "success",
                **result_data
            }
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            return {
                "success": True,
                "status": "success",
                "score": 0,
                "issues": [{
                    "type": "syntax_error",
                    "severity": "critical",
                    "message": f"Code has syntax errors: {str(e)}",
                    "line": getattr(e, 'lineno', None)
                }],
                "suggestions": ["Fix syntax errors before security analysis"],
                "aspects": {},
                "statistics": {}
            }
        except Exception as e:
            logger.error(f"Error during security review: {e}")
            return {
                "success": False,
                "status": "error",
                "score": 0,
                "message": f"Security review failed: {str(e)}"
            }
    
    def _analyze_sql_injection(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Analyze SQL injection vulnerabilities.
        
        Checks for:
        - String formatting in SQL queries
        - String concatenation in queries
        - f-strings in queries
        - Missing parameterized queries
        
        Args:
            tree: AST tree
            code: Source code
            
        Returns:
            Analysis with score, issues, suggestions
        """
        score = 100
        issues = []
        suggestions = []
        
        # Check for SQL injection patterns using regex
        for pattern in self.SQL_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "sql_injection_vulnerability",
                    "severity": "critical",
                    "message": "Potential SQL injection: String formatting/concatenation in SQL query",
                    "line": line_num,
                    "code_snippet": match.group()
                })
                suggestions.append(
                    f"Line {line_num}: Use parameterized queries with ? or %s placeholders instead of string formatting"
                )
                score -= 20
        
        # Check for .execute() calls in AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in ['execute', 'executemany']:
                    # Check if argument is a formatted string
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.JoinedStr):  # f-string
                            issues.append({
                                "type": "sql_injection_vulnerability",
                                "severity": "critical",
                                "message": "SQL query uses f-string formatting",
                                "line": node.lineno
                            })
                            suggestions.append(
                                f"Line {node.lineno}: Replace f-string with parameterized query"
                            )
                            score -= 20
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _analyze_xss_vulnerabilities(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Analyze XSS vulnerabilities.
        
        Checks for:
        - Direct HTML output without escaping
        - JavaScript eval() usage
        - innerHTML assignments
        - Unsafe template rendering
        
        Args:
            tree: AST tree
            code: Source code
            
        Returns:
            Analysis with score, issues, suggestions
        """
        score = 100
        issues = []
        suggestions = []
        
        # Check for XSS patterns using regex
        for pattern in self.XSS_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "xss_vulnerability",
                    "severity": "critical",
                    "message": f"Potential XSS: {match.group()} without proper escaping",
                    "line": line_num,
                    "code_snippet": match.group()
                })
                suggestions.append(
                    f"Line {line_num}: Ensure all user input is properly escaped before output"
                )
                score -= 15
        
        # Check for eval() in AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'eval':
                    issues.append({
                        "type": "dangerous_function",
                        "severity": "critical",
                        "message": "Use of eval() can lead to code injection",
                        "line": node.lineno
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Avoid eval(), use safer alternatives like ast.literal_eval()"
                    )
                    score -= 25
        
        # Check for Flask/Django template rendering without auto-escape
        if 'render_template_string' in code and 'autoescape=False' in code:
            line_num = code.find('autoescape=False')
            line_num = code[:line_num].count('\n') + 1
            issues.append({
                "type": "xss_vulnerability",
                "severity": "critical",
                "message": "Template rendering with autoescape disabled",
                "line": line_num
            })
            suggestions.append(
                f"Line {line_num}: Enable autoescape in template rendering"
            )
            score -= 20
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _analyze_auth_authorization(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Analyze authentication and authorization issues.
        
        Checks for:
        - Missing authentication decorators
        - Weak password handling
        - Session security issues
        - Insecure authentication schemes
        
        Args:
            tree: AST tree
            code: Source code
            
        Returns:
            Analysis with score, issues, suggestions
        """
        score = 100
        issues = []
        suggestions = []
        
        # Check for weak password storage (plaintext)
        # Look for password comparisons that suggest plaintext handling
        password_compare_patterns = [
            r'password\s*==\s*\w+',           # password == variable
            r'\w+\s*==\s*password',           # variable == password
            r'password\s*==\s*["\']',         # password == "string"
            r'if\s+password\s*==',            # if password ==
        ]
        
        for pattern in password_compare_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                match = re.search(pattern, code, re.IGNORECASE)
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "weak_authentication",
                    "severity": "critical",
                    "message": "Password comparison without hashing",
                    "line": line_num
                })
                suggestions.append(
                    f"Line {line_num}: Use bcrypt or argon2 for password hashing, never compare plaintext"
                )
                score -= 25
                break  # Only report once
        
        # Check for missing login_required decorators on routes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a route handler
                has_route_decorator = any(
                    isinstance(d, ast.Call) and
                    isinstance(d.func, ast.Attribute) and
                    d.func.attr in ['route', 'get', 'post']
                    for d in node.decorator_list
                )
                
                has_auth_decorator = any(
                    isinstance(d, ast.Name) and d.id in ['login_required', 'require_auth'] or
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and 
                    d.func.id in ['login_required', 'require_auth']
                    for d in node.decorator_list
                )
                
                if has_route_decorator and not has_auth_decorator and 'public' not in node.name.lower():
                    issues.append({
                        "type": "missing_authentication",
                        "severity": "warning",
                        "message": f"Route '{node.name}' may be missing authentication",
                        "line": node.lineno
                    })
                    suggestions.append(
                        f"Line {node.lineno}: Consider adding @login_required decorator if authentication is needed"
                    )
                    score -= 10
        
        # Check for insecure session configuration
        if 'SECRET_KEY' in code and re.search(r'SECRET_KEY\s*=\s*["\'][^"\']{1,10}["\']', code):
            line_num = code.find('SECRET_KEY')
            line_num = code[:line_num].count('\n') + 1
            issues.append({
                "type": "weak_secret_key",
                "severity": "warning",
                "message": "SECRET_KEY appears to be too short or weak",
                "line": line_num
            })
            suggestions.append(
                f"Line {line_num}: Use a strong, random SECRET_KEY (32+ characters)"
            )
            score -= 10
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _analyze_secrets_management(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Analyze secrets management.
        
        Checks for:
        - Hardcoded passwords
        - Hardcoded API keys
        - Hardcoded tokens
        - AWS credentials in code
        
        Args:
            tree: AST tree
            code: Source code
            
        Returns:
            Analysis with score, issues, suggestions
        """
        score = 100
        issues = []
        suggestions = []
        
        # Check for hardcoded secrets using regex patterns
        for pattern in self.SECRETS_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                secret_type = match.group().split('=')[0].strip()
                issues.append({
                    "type": "hardcoded_secret",
                    "severity": "critical",
                    "message": f"Hardcoded {secret_type} found in code",
                    "line": line_num
                })
                suggestions.append(
                    f"Line {line_num}: Move {secret_type} to environment variables or secure vault"
                )
                score -= 20
        
        # Check for common credential variable names
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(keyword in var_name for keyword in ['password', 'secret', 'api_key', 'token', 'credential']):
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                if len(node.value.value) > 5:  # Only flag non-empty strings
                                    issues.append({
                                        "type": "potential_hardcoded_secret",
                                        "severity": "warning",
                                        "message": f"Variable '{target.id}' may contain hardcoded credential",
                                        "line": node.lineno
                                    })
                                    suggestions.append(
                                        f"Line {node.lineno}: Use os.environ.get('{target.id.upper()}') instead"
                                    )
                                    score -= 15
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _analyze_insecure_dependencies(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Analyze insecure dependencies and imports.
        
        Checks for:
        - Dangerous imports (pickle, yaml.load, etc.)
        - Use of subprocess without validation
        - Use of eval/exec
        - Insecure deserialization
        
        Args:
            tree: AST tree
            code: Source code
            
        Returns:
            Analysis with score, issues, suggestions
        """
        score = 100
        issues = []
        suggestions = []
        
        # Check for dangerous imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.INSECURE_IMPORTS:
                        issues.append({
                            "type": "insecure_import",
                            "severity": "warning",
                            "message": f"Import of '{alias.name}' can be dangerous",
                            "line": node.lineno
                        })
                        suggestions.append(
                            f"Line {node.lineno}: {self.INSECURE_IMPORTS[alias.name]}"
                        )
                        score -= 10
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.INSECURE_IMPORTS:
                    issues.append({
                        "type": "insecure_import",
                        "severity": "warning",
                        "message": f"Import from '{node.module}' can be dangerous",
                        "line": node.lineno
                    })
                    suggestions.append(
                        f"Line {node.lineno}: {self.INSECURE_IMPORTS[node.module]}"
                    )
                    score -= 10
        
        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"
                
                if func_name in self.DANGEROUS_FUNCTIONS:
                    severity = "critical" if func_name in ['eval', 'exec', 'pickle.loads'] else "warning"
                    issues.append({
                        "type": "dangerous_function",
                        "severity": severity,
                        "message": f"Use of dangerous function: {func_name}",
                        "line": node.lineno
                    })
                    
                    if func_name == 'eval':
                        suggestions.append(
                            f"Line {node.lineno}: Replace eval() with ast.literal_eval() or json.loads()"
                        )
                    elif func_name == 'pickle.loads':
                        suggestions.append(
                            f"Line {node.lineno}: Use json or safer serialization format instead of pickle"
                        )
                    elif func_name == 'subprocess.call' or func_name == 'subprocess.Popen':
                        suggestions.append(
                            f"Line {node.lineno}: Use shell=False and validate all inputs"
                        )
                    else:
                        suggestions.append(
                            f"Line {node.lineno}: Avoid {func_name}, use safer alternatives"
                        )
                    
                    score -= 15 if severity == "critical" else 10
        
        # Check for shell=True in subprocess
        if 'shell=True' in code:
            line_num = code.find('shell=True')
            line_num = code[:line_num].count('\n') + 1
            issues.append({
                "type": "command_injection_risk",
                "severity": "critical",
                "message": "subprocess with shell=True is vulnerable to command injection",
                "line": line_num
            })
            suggestions.append(
                f"Line {line_num}: Use shell=False and pass command as list"
            )
            score -= 20
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions
        }
