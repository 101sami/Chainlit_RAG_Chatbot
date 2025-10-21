"""
Complete RAG-Powered Knowledge Assistant with Persistence
Includes document storage, conversation history, and ChatGPT-style interface
"""
import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import chainlit as cl
import openai
from dotenv import load_dotenv

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Document processing
import PyPDF2
from docx import Document as DocxDocument
from io import BytesIO

# Web scraping for Help Center integration
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse

# Load environment variables
load_dotenv()

class PersistentRAGAssistant:
    """Complete RAG Assistant with persistent storage"""

    def __init__(self):
        # OpenAI client configuration
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.specialization = os.getenv("PRODUCT_SPECIALIZATION", "Trend Micro")

        # Initialize data directories
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        self.conversations_dir = self.data_dir / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)
        self.documents_dir = self.data_dir / "documents"
        self.documents_dir.mkdir(exist_ok=True)

        # Initialize vector database
        self.chroma_client = None
        self.collection = None
        self.embeddings_model = None
        self.initialize_vector_db()

    def initialize_vector_db(self):
        """Initialize ChromaDB for persistent document storage"""
        try:
            # Initialize ChromaDB client
            chroma_path = self.data_dir / "chroma_db"
            chroma_path.mkdir(exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            collection_name = "trend_micro_knowledge"
            try:
                self.collection = self.chroma_client.get_collection(name=collection_name)
            except ValueError:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Knowledge base for {self.specialization}"}
                )

            # Initialize embeddings model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

            print(f"Vector database initialized: {self.collection.count()} documents loaded")

        except Exception as e:
            print(f"Error initializing vector database: {e}")

    def fetch_help_center_content(self, search_query: str, max_articles: int = 5) -> List[Dict[str, str]]:
        """Fetch relevant content from Trend Micro Help Center"""
        try:
            articles = []
            base_url = "https://helpcenter.trendmicro.com"

            # First try to search for articles
            search_url = f"{base_url}/eureka-search/"
            search_params = {
                'q': search_query,
                'p': 'all'  # Search all products
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Search for relevant articles
            response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for article links in search results
                article_links = soup.find_all('a', href=True)
                relevant_links = []

                for link in article_links:
                    href = link.get('href', '')
                    if '/en-us/' in href and any(keyword in href.lower() for keyword in ['article', 'solution', 'troubleshoot', 'install', 'support']):
                        if not href.startswith('http'):
                            href = base_url + href
                        relevant_links.append(href)

                # Fetch content from top articles
                for url in relevant_links[:max_articles]:
                    try:
                        time.sleep(1)  # Be respectful to the server
                        article_response = requests.get(url, headers=headers, timeout=10)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.content, 'html.parser')

                            # Extract title
                            title_elem = article_soup.find('h1') or article_soup.find('title')
                            title = title_elem.get_text(strip=True) if title_elem else "Trend Micro Help Article"

                            # Extract main content
                            content_selectors = [
                                '.article-content',
                                '.content-body',
                                '.help-content',
                                'main',
                                '.main-content',
                                'article'
                            ]

                            content = ""
                            for selector in content_selectors:
                                content_elem = article_soup.select_one(selector)
                                if content_elem:
                                    # Clean up the content
                                    for script in content_elem(["script", "style", "nav", "footer", "header"]):
                                        script.decompose()
                                    content = content_elem.get_text(separator='\n', strip=True)
                                    break

                            if content and len(content) > 100:  # Only add if substantial content
                                articles.append({
                                    'title': title,
                                    'content': content[:3000],  # Limit content length
                                    'url': url,
                                    'source': 'Trend Micro Help Center'
                                })

                    except Exception as e:
                        print(f"Error fetching article {url}: {e}")
                        continue

            # If no search results, try common help topics
            if not articles:
                common_topics = [
                    f"{base_url}/en-us/product-support/maximum-security/",
                    f"{base_url}/en-us/product-support/internet-security/",
                    f"{base_url}/en-us/product-support/antivirus-security/",
                    f"{base_url}/en-us/account-help/",
                    f"{base_url}/en-us/download-and-install/"
                ]

                for url in common_topics[:2]:  # Try first 2 common topics
                    try:
                        time.sleep(1)
                        response = requests.get(url, headers=headers, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')

                            title_elem = soup.find('h1') or soup.find('title')
                            title = title_elem.get_text(strip=True) if title_elem else "Trend Micro Help"

                            # Extract relevant content
                            content_elem = soup.find('main') or soup.find('.content') or soup.find('body')
                            if content_elem:
                                for script in content_elem(["script", "style", "nav", "footer", "header"]):
                                    script.decompose()
                                content = content_elem.get_text(separator='\n', strip=True)

                                if content and len(content) > 100:
                                    articles.append({
                                        'title': title,
                                        'content': content[:2000],
                                        'url': url,
                                        'source': 'Trend Micro Help Center'
                                    })

                    except Exception as e:
                        print(f"Error fetching common topic {url}: {e}")
                        continue

            return articles

        except Exception as e:
            print(f"Error fetching help center content: {e}")
            return []

    def add_help_center_to_kb(self, search_query: str) -> int:
        """Fetch and add Help Center content to knowledge base"""
        try:
            articles = self.fetch_help_center_content(search_query)
            added_count = 0

            for article in articles:
                # Create a combined content string
                full_content = f"Title: {article['title']}\n\nURL: {article['url']}\n\nContent:\n{article['content']}"

                # Use the URL as filename for uniqueness
                filename = f"helpcenter_{urllib.parse.quote(article['url'].split('/')[-1], safe='')}.txt"

                if self.add_document_to_kb(filename, full_content):
                    added_count += 1
                    print(f"Added Help Center article: {article['title']}")

            return added_count

        except Exception as e:
            print(f"Error adding help center content to KB: {e}")
            return 0

    def is_query_relevant(self, query: str, conversation_history: list = None) -> bool:
        """Check if query is relevant to Mobile Security and Home Network Security products (context-aware)"""
        query_lower = query.lower()

        # Keywords specific to Mobile Security and Home Network Security
        mobile_keywords = [
            "mobile security", "mobile", "phone", "smartphone", "android", "ios", "iphone", "ipad",
            "mobile app", "mobile protection", "mobile antivirus", "mobile threat", "mobile device",
            "tablet", "mobile malware", "mobile scan", "mobile firewall", "app security",
            "mobile vpn", "mobile privacy", "mobile backup", "parental control", "safe browsing"
        ]

        network_keywords = [
            "home network", "home security", "router", "wifi", "wireless", "network security",
            "home router", "router security", "wifi security", "network protection", "smart home",
            "iot security", "network monitoring", "bandwidth", "network firewall", "home network protection",
            "wireless security", "network threat", "router protection", "home wifi", "network scan"
        ]

        # Non-technical business/administrative keywords for our specialized products
        business_keywords = [
            "subscription", "renew", "renewal", "billing", "payment", "account", "license",
            "purchase", "buy", "pricing", "cost", "upgrade", "downgrade", "cancel", "refund",
            "activation", "activate", "register", "trial", "free trial", "support", "customer service"
        ]

        # Exclusion keywords that indicate non-mobile/non-home network topics
        exclusion_keywords = [
            "desktop", "laptop", "computer", "pc", "windows computer", "mac computer",
            "server", "enterprise", "business", "corporate", "office", "workstation",
            "endpoint", "cloud one", "deep security", "worry-free", "apex one"
        ]

        # First, check for exclusion keywords that indicate non-mobile/non-home network topics
        has_exclusion = any(keyword in query_lower for keyword in exclusion_keywords)
        if has_exclusion:
            print(f"DEBUG RELEVANCE: Found exclusion keyword, rejecting query")
            return False

        # Check if query mentions mobile security keywords
        has_mobile = any(keyword in query_lower for keyword in mobile_keywords)
        # Check if query mentions home network security keywords
        has_network = any(keyword in query_lower for keyword in network_keywords)
        # Check if query mentions business/administrative keywords
        has_business = any(keyword in query_lower for keyword in business_keywords)

        # Accept if direct mobile/network security topics
        if has_mobile or has_network:
            print(f"DEBUG RELEVANCE: Found mobile/network keyword, accepting query")
            return True

        # Accept business queries if they mention our specialized products
        if has_business:
            specialized_products = [
                "mobile security", "home network security", "home security", "network security"
            ]
            if any(product in query_lower for product in specialized_products):
                print(f"DEBUG RELEVANCE: Found business query related to specialized products, accepting")
                return True

        # Check conversation context for Mobile Security / Home Network Security relevance
        if conversation_history and len(conversation_history) > 0:
            print(f"DEBUG RELEVANCE: Checking context for relevance...")
            # Look at recent messages to see if we're in a relevant conversation
            recent_messages = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history

            for msg in recent_messages:
                content = msg.get('content', '').lower()
                # Check if recent context has exclusion keywords
                if any(keyword in content for keyword in exclusion_keywords):
                    print(f"DEBUG RELEVANCE: Found exclusion in context, rejecting")
                    return False
                # Check if recent context has mobile/network keywords
                if (any(keyword in content for keyword in mobile_keywords) or
                    any(keyword in content for keyword in network_keywords)):
                    print(f"DEBUG RELEVANCE: Found mobile/network context in: {content[:50]}...")
                    return True
                # Check if recent context has business keywords related to our products
                if any(keyword in content for keyword in business_keywords):
                    specialized_products = [
                        "mobile security", "home network security", "home security", "network security"
                    ]
                    if any(product in content for product in specialized_products):
                        print(f"DEBUG RELEVANCE: Found business context related to specialized products")
                        return True

        # Check for contextual pronouns that suggest continuation (only if previous context was mobile/network)
        contextual_words = ["it", "this", "that", "them", "they"]
        if any(word in query_lower for word in contextual_words) and conversation_history:
            # Only allow contextual references if recent context was about mobile/network/business security
            recent_messages = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
            for msg in recent_messages:
                content = msg.get('content', '').lower()
                if (any(keyword in content for keyword in mobile_keywords) or
                    any(keyword in content for keyword in network_keywords)):
                    print(f"DEBUG RELEVANCE: Found contextual reference with mobile/network context, allowing query")
                    return True
                # Also allow if context was business-related to our specialized products
                if any(keyword in content for keyword in business_keywords):
                    specialized_products = [
                        "mobile security", "home network security", "home security", "network security"
                    ]
                    if any(product in content for product in specialized_products):
                        print(f"DEBUG RELEVANCE: Found contextual reference with business context, allowing query")
                        return True

        return False

    def analyze_query_specificity(self, query: str, conversation_history: list = None) -> Dict[str, Any]:
        """Analyze if a query is vague and needs probing questions"""
        query_lower = query.lower()

        # Check recent conversation history for context that might make the query less vague
        context_details = set()
        if conversation_history:
            # Look at last 4 messages for context
            recent_messages = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
            for msg in recent_messages:
                content = msg.get('content', '').lower()
                # Extract context details from recent messages
                if any(detail in content for detail in ['windows', 'mac', 'android', 'ios']):
                    context_details.add('device_type')
                if any(detail in content for detail in ['error code', 'error message', 'code']):
                    context_details.add('error_details')
                if any(detail in content for detail in ['maximum security', 'internet security', 'antivirus']):
                    context_details.add('product_info')
                if any(detail in content for detail in ['version', 'build', 'update']):
                    context_details.add('version_info')

        # Define vague patterns that need clarification
        vague_patterns = {
            'error_codes': {
                'patterns': ['getting an error', 'shows error', 'error message', 'having an error', 'see an error', 'encountering an error', 'strange error', 'weird error'],
                'missing_info': ['specific error code', 'device type', 'when the error occurs'],
                'probing_questions': [
                    "What's the **exact error code or message** you're seeing?",
                    "What **device** are you using? (Windows, Mac, Android, or iOS)",
                    "**When** does this error appear? (during installation, scanning, startup, etc.)",
                    "Which **Trend Micro product** are you using?"
                ]
            },
            'installation_issues': {
                'patterns': ['wont install', "can't install", 'install problem', 'installation failed', 'setup failed', 'installation not working', 'having trouble installing'],
                'missing_info': ['operating system', 'installation stage', 'specific error'],
                'probing_questions': [
                    "What **operating system** are you using? (Windows 10/11, macOS, etc.)",
                    "At what **stage** does the installation fail? (download, setup, activation)",
                    "Are you getting any **specific error messages**?",
                    "Is this your **first attempt** or have you tried before?"
                ]
            },
            'performance_issues': {
                'patterns': ['running slow', 'really slow', 'freezing', 'crashes', 'not working properly', 'performance issues', 'having performance problems'],
                'missing_info': ['specific symptoms', 'when it happens', 'system specs'],
                'probing_questions': [
                    "What **specific symptoms** are you experiencing? (slow scanning, freezes, crashes)",
                    "**When** does this happen? (during scans, startup, continuously)",
                    "What are your **system specs**? (RAM, processor, storage)",
                    "How **long** has this been happening?"
                ]
            },
            'scanning_issues': {
                'patterns': ['scan not working', 'scan problem', 'scanning issues', 'scan failed', 'scan stops', 'having trouble scanning'],
                'missing_info': ['scan type', 'specific issue', 'scan results'],
                'probing_questions': [
                    "What **type of scan** are you trying? (Quick, Full, Custom)",
                    "What **exactly** happens when you scan? (won't start, stops, no results)",
                    "Are you getting any **error messages** during scanning?",
                    "What **area** are you scanning? (whole computer, specific drive, folders)"
                ]
            },
            'activation_issues': {
                'patterns': ['activation problem', 'activation failed', 'license problem', 'product key not working', 'subscription issues', 'having trouble activating'],
                'missing_info': ['activation method', 'error details', 'purchase info'],
                'probing_questions': [
                    "How are you trying to **activate**? (product key, online account, subscription)",
                    "What **error message** appears during activation?",
                    "Is this a **new purchase** or renewal?",
                    "Which **Trend Micro product** are you activating?"
                ]
            },
            'general_help': {
                'patterns': ['having a problem', 'having an issue', 'having trouble', 'something is wrong', 'not working properly', 'need help with something'],
                'missing_info': ['specific issue', 'product details', 'what you tried'],
                'probing_questions': [
                    "What **specific issue** are you experiencing?",
                    "Which **Trend Micro product** are you using?",
                    "What **troubleshooting steps** have you tried?",
                    "**When** did this problem start?"
                ]
            }
        }

        # Check if query matches vague patterns
        for category, data in vague_patterns.items():
            for pattern in data['patterns']:
                if pattern in query_lower:
                    # Check if query is too short or lacks specific details
                    word_count = len(query.split())
                    has_specifics = any(detail in query_lower for detail in [
                        'code', 'message', 'windows', 'mac', 'android', 'ios',
                        'maximum security', 'internet security', 'antivirus',
                        'version', 'build', 'exactly', 'specifically'
                    ])

                    # Determine if we need to probe
                    needs_probing = (
                        word_count < 8 or  # Very short query
                        not has_specifics or  # No specific details mentioned
                        query_lower.count('error') > 0 and 'code' not in query_lower  # Error mentioned but no code
                    )

                    if needs_probing:
                        # Filter out questions that are already answered by context
                        filtered_questions = []
                        for question in data['probing_questions']:
                            question_lower = question.lower()
                            skip_question = False

                            # Skip device questions if we already know the device
                            if 'device' in question_lower and 'device_type' in context_details:
                                skip_question = True
                            # Skip error questions if we already have error details
                            elif 'error' in question_lower and 'error_details' in context_details:
                                skip_question = True
                            # Skip product questions if we already know the product
                            elif 'product' in question_lower and 'product_info' in context_details:
                                skip_question = True

                            if not skip_question:
                                filtered_questions.append(question)

                        # Only return probing questions if we still have questions to ask
                        if filtered_questions:
                            return {
                                'is_vague': True,
                                'category': category,
                                'missing_info': data['missing_info'],
                                'probing_questions': filtered_questions,
                                'confidence': 0.8
                            }
                        else:
                            # All context is available, proceed with normal response
                            return {'is_vague': False, 'confidence': 0.9}

        return {'is_vague': False, 'confidence': 0.9}

    def generate_probing_response(self, analysis: Dict[str, Any], original_query: str) -> str:
        """Generate a response with probing questions"""
        category = analysis['category']
        questions = analysis['probing_questions']

        category_names = {
            'error_codes': 'error',
            'installation_issues': 'installation issue',
            'performance_issues': 'performance problem',
            'scanning_issues': 'scanning issue',
            'activation_issues': 'activation problem',
            'general_help': 'request'
        }

        empathy_starters = {
            'error_codes': "I understand error messages can be frustrating. Let me help you figure this out.",
            'installation_issues': "I'll help you get this installation working properly.",
            'performance_issues': "Let's figure out what's causing these performance issues.",
            'scanning_issues': "I'll help you get your scanning working correctly.",
            'activation_issues': "Let me help you get your product activated.",
            'general_help': "I'm here to help you with this!"
        }

        response = f"""{empathy_starters.get(category, "I'm here to help you with this!")}

To give you the best possible solution for your {category_names.get(category, 'situation')}, I'd like to understand a bit more about what's happening:

"""

        # Add numbered probing questions
        for i, question in enumerate(questions, 1):
            response += f"**{i}.** {question}\n\n"

        response += f"""---

You don't need to answer everything - even partial information helps! I'll search our knowledge base and Trend Micro's Help Center for the best solution.

**Feel free to provide any details you have, and I'll help you from there.**"""

        return response

    def is_follow_up_response(self, query: str, conversation_history: list = None) -> bool:
        """Detect if the current query is a follow-up response to probing questions"""
        if not conversation_history or len(conversation_history) < 2:
            return False

        # Check if the last assistant message contained probing questions
        last_assistant_msg = None
        for msg in reversed(conversation_history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg
                break

        if not last_assistant_msg:
            return False

        # Check if the last assistant message had probing questions
        assistant_content = last_assistant_msg.get('content', '').lower()
        has_probing_indicators = any(indicator in assistant_content for indicator in [
            'need more information',
            'what is the exact',
            'what device',
            'what operating system',
            'which trend micro product',
            'what error code',
            'when does this',
            'what specific'
        ])

        # Check if current query seems like an answer (contains specific details)
        query_lower = query.lower()
        has_specific_details = any(detail in query_lower for detail in [
            'windows', 'mac', 'android', 'ios', 'error code', 'code', 'message',
            'maximum security', 'internet security', 'antivirus', 'version',
            'during', 'when', 'while', 'it happens', 'occurring'
        ])

        return has_probing_indicators and has_specific_details

    def process_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text_content = []

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")

            return "\n\n".join(text_content)
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def process_docx(self, uploaded_file) -> str:
        """Extract text from Word document"""
        try:
            doc = DocxDocument(BytesIO(uploaded_file.read()))
            content_parts = []

            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    content_parts.append(text)

            return "\n\n".join(content_parts)
        except Exception as e:
            return f"Error processing Word document: {str(e)}"

    def process_text_file(self, uploaded_file) -> str:
        """Process plain text file"""
        try:
            return uploaded_file.read().decode('utf-8')
        except Exception as e:
            return f"Error processing text file: {str(e)}"

    def add_document_to_kb(self, filename: str, content: str) -> bool:
        """Add document to persistent knowledge base"""
        try:
            if not self.collection or not self.embeddings_model:
                return False

            # Create document hash for deduplication
            doc_hash = hashlib.md5(content.encode()).hexdigest()

            # Check if document already exists
            existing_docs = self.collection.get(where={"doc_hash": doc_hash})
            if existing_docs['ids']:
                return False  # Document already exists

            # Split content into chunks
            chunk_size = 1000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

            # Generate embeddings and store
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_hash}_chunk_{i}"
                embedding = self.embeddings_model.encode([chunk]).tolist()[0]

                self.collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "filename": filename,
                        "doc_hash": doc_hash,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "uploaded_at": datetime.now().isoformat()
                    }]
                )

            # Save document metadata
            doc_metadata = {
                "filename": filename,
                "doc_hash": doc_hash,
                "content_length": len(content),
                "chunk_count": len(chunks),
                "uploaded_at": datetime.now().isoformat()
            }

            metadata_file = self.documents_dir / f"{doc_hash}.json"
            with open(metadata_file, 'w') as f:
                json.dump(doc_metadata, f, indent=2)

            return True

        except Exception as e:
            print(f"Error adding document to knowledge base: {e}")
            return False

    def search_knowledge_base(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant documents"""
        try:
            if not self.collection or not self.embeddings_model:
                return []

            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query]).tolist()[0]

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )

            # Process results
            relevant_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = 1 - distance
                    relevant_docs.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity': similarity,
                        'rank': i + 1
                    })

            return relevant_docs

        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []

    def get_response_sync(self, user_query: str, conversation_history: list = None) -> str:
        """Generate context-aware response using RAG pipeline (synchronous version)"""
        try:
            # Check relevance with conversation context
            if not self.is_query_relevant(user_query, conversation_history):
                return f"""I'm sorry, but I'm specialized in **{self.specialization}** products and services.
Your question appears to be outside my area of expertise.

Please ask me questions related to:
- 📱 Mobile Security (Android/iOS protection, mobile apps, mobile threats)
- 🏠 Home Network Security (router security, WiFi protection, IoT security)
- ⚙️ Installation and setup guides
- 🔧 Troubleshooting and support
- 📋 Best practices and recommendations
- 🔑 Licensing and activation

How can I help you with {self.specialization} today?"""

            # Check if query is vague and needs probing questions
            specificity_analysis = self.analyze_query_specificity(user_query, conversation_history)
            if specificity_analysis['is_vague']:
                return self.generate_probing_response(specificity_analysis, user_query)

            # Search knowledge base
            relevant_docs = self.search_knowledge_base(user_query)

            # Prepare context from retrieved documents
            context = ""
            if relevant_docs:
                context = "\n\nRelevant information from knowledge base:\n"
                for i, doc in enumerate(relevant_docs, 1):
                    filename = doc['metadata'].get('filename', 'Unknown')
                    similarity = doc['similarity']
                    content = doc['content']
                    context += f"\n**Source {i}: {filename}** (Relevance: {similarity:.0%})\n{content}\n"
            else:
                # No relevant documents found - AI should not make up information
                context = "\n\n⚠️ **IMPORTANT: No relevant information found in knowledge base for this query. DO NOT make up or assume any information.**\n"

            # Prepare conversation context
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                conversation_context = "\n\nPrevious conversation context:\n"
                # Include last 6 messages for context (3 exchanges)
                recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200] + "..." if len(msg.get('content', '')) > 200 else msg.get('content', '')
                    conversation_context += f"{role.title()}: {content}\n"

            # Prepare system prompt with context awareness
            system_prompt = f"""You are a friendly, empathetic, and knowledgeable support specialist for {self.specialization} products and services. You communicate like a human support agent who genuinely cares about helping customers solve their problems.

Your personality and approach:
- Warm, conversational, and understanding
- Show empathy when users are frustrated or stressed
- Use natural language, not robotic responses
- Acknowledge the user's feelings and situation
- Be encouraging and reassuring

You help with:
- Product features and capabilities
- Installation and configuration
- Troubleshooting issues
- Best practices and recommendations
- Licensing and activation
- System requirements
- Security policies and settings

IMPORTANT - Human-like Communication:
1. Always acknowledge the user's emotions and situation with empathy
2. Use conversational language like "I understand how frustrating this must be"
3. Reference previous conversation naturally (e.g., "Like we talked about earlier...", "Building on what you mentioned...")
4. If the user refers to "it", "that", "the previous solution", use conversation context seamlessly
5. Maintain a helpful, patient tone throughout the conversation
6. Remember the user's specific situation and refer back to it

Communication Guidelines:
1. Start responses with empathy when appropriate
2. **CRITICAL: ONLY use information from the provided knowledge base context or Trend Micro Help Center**
3. **NEVER make up, assume, or hallucinate information about error codes, solutions, or procedures**
4. **ERROR CODES: If a user mentions an error code, ONLY provide information if it exists in the knowledge base. If not found, say "I don't have information about that specific error code in our knowledge base."**
5. If you don't have specific information in the knowledge base, clearly state: "I don't have specific information about that in our knowledge base. Let me search the Help Center for you using the /fetch command."
5. Focus on {self.specialization} products with expertise and confidence
6. Always cite sources when using information from documents or Help Center articles
7. Use encouraging language but stay factual and accurate
8. For basic business processes (renewals, billing, account management) related to our specialized products, you may provide general guidance while offering to fetch specific Help Center information
9. If asked about something not in the knowledge base, offer to fetch current Help Center information instead of guessing

{context}{conversation_context}"""

            # Build message history for the API call
            messages = [{"role": "system", "content": system_prompt}]

            # Add recent conversation history to maintain context
            if conversation_history:
                # Include last 8 messages (4 exchanges) for better context
                recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
                for msg in recent_history:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })

            # Add current user query
            messages.append({"role": "user", "content": user_query})

            # Create chat completion with conversation context
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            return f"""❌ **Error Processing Request**

I encountered an error while processing your question:

**Error Details:** `{error_msg}`

**Possible Solutions:**
- Check your internet connection
- Verify API configuration in the .env file
- Try rephrasing your question
- Contact your administrator if the issue persists"""

    async def get_response(self, user_query: str, conversation_history: list = None) -> str:
        """Generate context-aware response using RAG pipeline and conversation history"""
        try:
            print(f"DEBUG RAG: get_response called with query: {user_query[:50]}...")
            print(f"DEBUG RAG: Conversation history length: {len(conversation_history) if conversation_history else 0}")

            # Check relevance with conversation context
            if not self.is_query_relevant(user_query, conversation_history):
                return f"""I'm sorry, but I'm specialized in **{self.specialization}** products and services.
Your question appears to be outside my area of expertise.

Please ask me questions related to:
- 📱 Mobile Security (Android/iOS protection, mobile apps, mobile threats)
- 🏠 Home Network Security (router security, WiFi protection, IoT security)
- ⚙️ Installation and setup guides
- 🔧 Troubleshooting and support
- 📋 Best practices and recommendations
- 🔑 Licensing and activation

How can I help you with {self.specialization} today?"""

            # Check if this is a follow-up response to probing questions
            is_followup = self.is_follow_up_response(user_query, conversation_history)
            if is_followup:
                print("DEBUG RAG: Detected follow-up response with more details")
                # Add acknowledgment prefix to the normal response
                acknowledgments = [
                    "Perfect! Thanks for those details - that really helps me understand what's going on. Let me find the best solution for you.\n\n",
                    "Great! With that information, I can give you much more targeted help. Here's what I found:\n\n",
                    "Excellent! Those details are exactly what I needed. Let me search for the most relevant solution:\n\n",
                    "Thank you for the additional information! Now I can provide you with a much more specific solution:\n\n"
                ]
                import random
                followup_prefix = random.choice(acknowledgments)
            else:
                followup_prefix = ""

            # Check if query is vague and needs probing questions (only if not a follow-up)
            if not is_followup:
                specificity_analysis = self.analyze_query_specificity(user_query, conversation_history)
                if specificity_analysis['is_vague']:
                    print(f"DEBUG RAG: Query is vague, generating probing questions for category: {specificity_analysis['category']}")
                    return self.generate_probing_response(specificity_analysis, user_query)

            # Search knowledge base first
            relevant_docs = self.search_knowledge_base(user_query)

            # If no relevant documents found in local KB, try Help Center
            if not relevant_docs or len(relevant_docs) < 2:
                print(f"DEBUG RAG: Limited local content, fetching from Help Center...")
                try:
                    help_articles = self.fetch_help_center_content(user_query, max_articles=3)
                    if help_articles:
                        # Add fetched articles to the context
                        for article in help_articles:
                            relevant_docs.append({
                                'content': article['content'],
                                'metadata': {
                                    'filename': f"Help Center: {article['title']}",
                                    'source': article['url']
                                },
                                'similarity': 0.95,  # High relevance since it's live fetched
                                'rank': len(relevant_docs) + 1
                            })
                        print(f"DEBUG RAG: Added {len(help_articles)} Help Center articles to context")
                except Exception as e:
                    print(f"DEBUG RAG: Error fetching Help Center content: {e}")

            # Prepare context from retrieved documents
            context = ""
            if relevant_docs:
                context = "\n\nRelevant information from knowledge base:\n"
                for i, doc in enumerate(relevant_docs, 1):
                    filename = doc['metadata'].get('filename', 'Unknown')
                    similarity = doc['similarity']
                    content = doc['content']
                    context += f"\n**Source {i}: {filename}** (Relevance: {similarity:.0%})\n{content}\n"
            else:
                # No relevant documents found - AI should not make up information
                context = "\n\n⚠️ **IMPORTANT: No relevant information found in knowledge base for this query. DO NOT make up or assume any information.**\n"

            # Prepare conversation context
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                print(f"DEBUG RAG: Processing {len(conversation_history)} messages for context")
                conversation_context = "\n\nPrevious conversation context:\n"
                # Include last 6 messages for context (3 exchanges)
                recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200] + "..." if len(msg.get('content', '')) > 200 else msg.get('content', '')
                    conversation_context += f"{role.title()}: {content}\n"
                    print(f"DEBUG RAG: Added to context - {role}: {content[:50]}...")
            else:
                print("DEBUG RAG: No conversation history provided")

            # Prepare system prompt with context awareness
            system_prompt = f"""You are a friendly, empathetic, and knowledgeable support specialist for {self.specialization} products and services. You communicate like a human support agent who genuinely cares about helping customers solve their problems.

Your personality and approach:
- Warm, conversational, and understanding
- Show empathy when users are frustrated or stressed
- Use natural language, not robotic responses
- Acknowledge the user's feelings and situation
- Be encouraging and reassuring

You help with:
- Product features and capabilities
- Installation and configuration
- Troubleshooting issues
- Best practices and recommendations
- Licensing and activation
- System requirements
- Security policies and settings

IMPORTANT - Human-like Communication:
1. Always acknowledge the user's emotions and situation with empathy
2. Use conversational language like "I understand how frustrating this must be"
3. Reference previous conversation naturally (e.g., "Like we talked about earlier...", "Building on what you mentioned...")
4. If the user refers to "it", "that", "the previous solution", use conversation context seamlessly
5. Maintain a helpful, patient tone throughout the conversation
6. Remember the user's specific situation and refer back to it

Communication Guidelines:
1. Start responses with empathy when appropriate
2. **CRITICAL: ONLY use information from the provided knowledge base context or Trend Micro Help Center**
3. **NEVER make up, assume, or hallucinate information about error codes, solutions, or procedures**
4. **ERROR CODES: If a user mentions an error code, ONLY provide information if it exists in the knowledge base. If not found, say "I don't have information about that specific error code in our knowledge base."**
5. If you don't have specific information in the knowledge base, clearly state: "I don't have specific information about that in our knowledge base. Let me search the Help Center for you using the /fetch command."
5. Focus on {self.specialization} products with expertise and confidence
6. Always cite sources when using information from documents or Help Center articles
7. Use encouraging language but stay factual and accurate
8. For basic business processes (renewals, billing, account management) related to our specialized products, you may provide general guidance while offering to fetch specific Help Center information
9. If asked about something not in the knowledge base, offer to fetch current Help Center information instead of guessing

{context}{conversation_context}"""

            # Build message history for the API call
            messages = [{"role": "system", "content": system_prompt}]

            # Add recent conversation history to maintain context
            if conversation_history:
                # Include last 8 messages (4 exchanges) for better context
                recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
                for msg in recent_history:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })

            # Add current user query
            messages.append({"role": "user", "content": user_query})

            # Create chat completion with conversation context
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )

            final_response = response.choices[0].message.content
            return followup_prefix + final_response

        except Exception as e:
            error_msg = str(e)
            return f"""❌ **Error Processing Request**

I encountered an error while processing your question:

**Error Details:** `{error_msg}`

**Possible Solutions:**
- Check your internet connection
- Verify API configuration in the .env file
- Try rephrasing your question
- Contact your administrator if the issue persists"""

    def save_conversation(self, conversation_id: str, messages: List[Dict], title: str = None):
        """Save conversation to persistent storage"""
        try:
            if not title:
                # Generate title from first user message
                user_messages = [msg for msg in messages if msg.get('role') == 'user']
                if user_messages:
                    title = user_messages[0]['content'][:50] + "..." if len(user_messages[0]['content']) > 50 else user_messages[0]['content']
                else:
                    title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            conversation_data = {
                "id": conversation_id,
                "title": title,
                "messages": messages,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "message_count": len(messages)
            }

            conversation_file = self.conversations_dir / f"{conversation_id}.json"
            with open(conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)

        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Load conversation from persistent storage"""
        try:
            conversation_file = self.conversations_dir / f"{conversation_id}.json"
            if conversation_file.exists():
                with open(conversation_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None

    def list_conversations(self) -> List[Dict]:
        """List all saved conversations"""
        try:
            conversations = []
            for file_path in self.conversations_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        conv_data = json.load(f)
                        conversations.append({
                            "id": conv_data.get("id"),
                            "title": conv_data.get("title"),
                            "last_updated": conv_data.get("last_updated"),
                            "message_count": conv_data.get("message_count", 0)
                        })
                except Exception as e:
                    print(f"Error reading conversation file {file_path}: {e}")

            # Sort by last updated, most recent first
            conversations.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            return conversations

        except Exception as e:
            print(f"Error listing conversations: {e}")
            return []

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            if not self.collection:
                return {"document_count": 0, "chunk_count": 0, "specialization": self.specialization}

            chunk_count = self.collection.count()

            # Count unique documents
            all_docs = self.collection.get()
            unique_docs = set()
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    if 'doc_hash' in metadata:
                        unique_docs.add(metadata['doc_hash'])

            return {
                "document_count": len(unique_docs),
                "chunk_count": chunk_count,
                "specialization": self.specialization
            }

        except Exception as e:
            print(f"Error getting knowledge base stats: {e}")
            return {"document_count": 0, "chunk_count": 0, "specialization": self.specialization}

# Global assistant instance
assistant = PersistentRAGAssistant()

# Session state for current conversation
current_conversation_id = None
current_messages = []