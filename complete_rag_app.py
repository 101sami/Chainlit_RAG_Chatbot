"""
Complete RAG-Powered Knowledge Assistant with ChatGPT-style Interface
Features: Document upload, persistent storage, conversation history, and full RAG pipeline
"""
import uuid
from datetime import datetime
import chainlit as cl
from rag_assistant import PersistentRAGAssistant

# Global assistant instance
assistant = PersistentRAGAssistant()

@cl.on_chat_start
async def start():
    """Initialize chat session with conversation history"""

    # Inject custom CSS for beautiful UI
    await cl.Html(
        content="""
        <link rel="stylesheet" href="/public/custom.css">
        <style>
        /* Additional inline styles for immediate effect */
        body, .chainlit-app {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        </style>
        """,
        display="inline"
    ).send()

    # Generate new conversation ID
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)
    cl.user_session.set("messages", [])

    # Check API configuration
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        await cl.Message(
            content="❌ **Configuration Error**: OPENAI_API_KEY not found in .env file. Please configure your Trend Micro AI Endpoint credentials.",
            author="System"
        ).send()
        return

    # Get knowledge base stats
    kb_stats = assistant.get_knowledge_base_stats()

    # Get conversation history for sidebar
    conversations = assistant.list_conversations()

    # Create sidebar with commands, conversation history, and settings
    settings = [
        cl.input_widget.TextInput(
            id="quick_commands",
            label="📝 Quick Commands",
            placeholder="Type these commands in chat:",
            description="/update - Check status\n/upload - Add documents\n/reset - Clear history\n/help - Get help",
            disabled=True
        )
    ]

    # Add conversation history to sidebar if available
    if conversations:
        # Create dropdown with conversation titles for easy selection
        conversation_options = []
        conversation_mapping = {}

        for i, conv in enumerate(conversations[:15]):  # Show last 15 conversations
            title = conv.get('title', 'Untitled Conversation')
            date = conv.get('last_updated', 'Unknown')[:10]
            msg_count = conv.get('message_count', 0)

            # Create user-friendly display text
            display_text = f"{title[:30]}{'...' if len(title) > 30 else ''} ({date}, {msg_count} msgs)"
            conversation_options.append(display_text)
            conversation_mapping[display_text] = conv.get('id', '')

        settings.append(
            cl.input_widget.Select(
                id="conversation_selector",
                label="💬 Conversation History",
                values=conversation_options,
                initial_index=0,
                description="Select a conversation to load. Shows title, date, and message count for easy identification."
            )
        )

        # Store mapping in session for later use
        cl.user_session.set("conversation_mapping", conversation_mapping)

        settings.append(
            cl.input_widget.TextInput(
                id="load_instruction",
                label="🔄 How to Load",
                placeholder="Select conversation above, then type /load selected",
                description="1. Choose conversation from dropdown above\n2. Type '/load selected' to load it\n3. Or use '/load [partial-title]' with conversation title",
                disabled=True
            )
        )
    else:
        settings.append(
            cl.input_widget.TextInput(
                id="no_history",
                label="💬 Conversation History",
                placeholder="No conversations yet",
                description="Start chatting to create conversation history. Previous conversations will appear here for easy access.",
                disabled=True
            )
        )

    # Add settings
    settings.extend([
        cl.input_widget.Switch(
            id="show_sources",
            label="📊 Show Document Sources",
            initial=True,
            description="Display which documents were used in responses"
        ),
        cl.input_widget.Slider(
            id="max_docs",
            label="📄 Max Documents to Search",
            initial=5,
            min=1,
            max=10,
            step=1,
            description="Maximum number of documents to retrieve"
        )
    ])

    await cl.ChatSettings(settings).send()


    # Welcome message with knowledge base info
    await cl.Message(
        content=f"""# 🛡️ Welcome to Trend Micro Knowledge Assistant!

I'm your specialized AI assistant for **{assistant.specialization}** products and services.

## 📊 Knowledge Base Status:
- **Documents**: {kb_stats['document_count']} documents loaded
- **Chunks**: {kb_stats['chunk_count']} searchable chunks
- **Specialization**: {kb_stats['specialization']}

## 🚀 What I can help you with:
- 🛡️ Product features and capabilities
- ⚙️ Installation and configuration guides
- 🔧 Troubleshooting and technical support
- 📋 Best practices and recommendations
- 🔑 Licensing and activation
- 🖥️ System requirements

**What would you like to know about {assistant.specialization}?**""",
        author="Assistant"
    ).send()

    # Create a prominent commands reference panel
    await cl.Message(
        content="""# 🎛️ **Available Commands & Features**

## 📝 **Quick Commands** (Type these in the chat):

### 🔄 `/update` or `/stats`
**Check Knowledge Base Status**
- View total documents and chunks loaded
- See system configuration and statistics
- Check if knowledge base is operational

### 📁 `/upload`
**Get Document Upload Instructions**
- Learn how to add PDF, DOCX, TXT, MD files
- Understand supported file formats
- See upload guidelines and best practices

### 🌐 `/fetch [topic]`
**Fetch Help Center Content**
- Get latest information from Trend Micro Help Center
- Add fresh content to knowledge base
- Example: `/fetch installation` or `/fetch troubleshooting`

### 💬 `/history`
**View Conversation History**
- See all your previous conversations
- View conversation titles and message counts
- Check timestamps of past chats

### 🗑️ `/reset`
**Reset/Clear Conversations**
- Start a fresh conversation
- Clear current chat history
- Save current conversation before resetting

### 🆘 `/help`
**Complete Help Guide**
- View all available commands
- Get detailed usage instructions
- See example questions and tips

---

## 📎 **File Upload Feature**
**Click the attachment button (📎) in the message input to:**
- Upload PDF documents (manuals, guides)
- Add Word documents (.docx)
- Include text files (.txt, .md)
- Expand the knowledge base with your documents

---

## ⚙️ **Settings Panel**
**Use the sidebar settings to:**
- Toggle document source display on/off
- Adjust maximum documents to search (1-10)
- Customize your experience

---

💡 **Pro Tip:** Just type any command (like `/stats`) or ask questions about Trend Micro products!""",
        author="📋 Command Reference",
        elements=[]
    ).send()

async def show_persistent_control_panel():
    """Create a persistent control panel message with action buttons"""
    try:
        # Create action buttons
        actions = [
            cl.Action(
                name="update_knowledge",
                value="update",
                label="🔄 Update Knowledge",
                description="Check knowledge base status"
            ),
            cl.Action(
                name="add_resources",
                value="add_resources",
                label="📁 Add Resources",
                description="Upload new documents"
            ),
            cl.Action(
                name="reset_conversations",
                value="reset",
                label="🗑️ Reset Conversations",
                description="Clear conversation history"
            ),
            cl.Action(
                name="view_history",
                value="history",
                label="💬 View History",
                description="Show conversation history"
            ),
            cl.Action(
                name="view_stats",
                value="stats",
                label="📊 View Stats",
                description="System statistics"
            ),
            cl.Action(
                name="new_conversation",
                value="new",
                label="✨ New Conversation",
                description="Start fresh conversation"
            )
        ]

        # Get conversation history for display
        conversations = assistant.list_conversations()

        # Create control panel content
        panel_content = """# 🎛️ Control Panel

## 🔧 Quick Actions
Click any button below to perform common tasks:

**Available Actions:**
- 🔄 **Update Knowledge** - Check knowledge base status
- 📁 **Add Resources** - Upload documents to knowledge base
- 🗑️ **Reset Conversations** - Clear all conversation history
- 💬 **View History** - Display conversation history
- 📊 **View Stats** - Show system statistics
- ✨ **New Conversation** - Start a fresh conversation

---

## 💬 Recent Conversations"""

        if conversations:
            panel_content += "\n\n"
            for i, conv in enumerate(conversations[:5], 1):
                title = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                panel_content += f"**{i}.** {title}\n"
                panel_content += f"   📅 {conv['last_updated'][:10]} • 💬 {conv['message_count']} messages\n\n"
        else:
            panel_content += "\n\nNo previous conversations found. Start chatting to create your first conversation!"

        panel_content += "\n---\n\n💡 **Tip:** You can also use commands like `/help`, `/stats`, `/history`, `/reset`"

        # Send the control panel as a pinned message with actions
        await cl.Message(
            content=panel_content,
            author="🎛️ Control Panel",
            actions=actions
        ).send()

        # Create minimal settings for document preferences
        settings = [
            cl.input_widget.Switch(
                id="show_sources",
                label="📊 Show Document Sources",
                initial=True,
                description="Display source documents in responses"
            ),
            cl.input_widget.Slider(
                id="max_docs",
                label="📄 Max Documents",
                initial=5,
                min=1,
                max=10,
                step=1,
                description="Maximum documents to search"
            )
        ]

        await cl.ChatSettings(settings).send()

    except Exception as e:
        print(f"Error creating control panel: {e}")

@cl.on_settings_update
async def setup_agent(settings):
    """Handle sidebar settings updates"""
    try:
        # Store settings in session for use in responses
        cl.user_session.set("settings", settings)

    except Exception as e:
        print(f"Error handling settings update: {e}")

@cl.action_callback("update_knowledge")
async def on_action_update(action):
    """Handle update knowledge base action"""
    kb_stats = assistant.get_knowledge_base_stats()
    await cl.Message(
        content=f"""🔄 **Knowledge Base Update Status**

## 📊 Current Statistics:
- **Documents**: {kb_stats['document_count']} files
- **Chunks**: {kb_stats['chunk_count']} searchable segments
- **Status**: ✅ Operational

## 🔄 Update Options:
1. **Add New Documents**: Use the 📎 attachment button to upload files
2. **Refresh Embeddings**: Knowledge base is automatically optimized
3. **Check Status**: Use `/stats` command for detailed information

**The knowledge base is persistent and automatically updated when you add new documents!**""",
        author="System"
    ).send()

@cl.action_callback("add_resources")
async def on_action_add_resources(action):
    """Handle add resources action"""
    await cl.Message(
        content="""📁 **Add Resources to Knowledge Base**

## 🔄 How to Upload Documents:

1. **Click the attachment button** (📎) in the message input area
2. **Select your files** - supports multiple files at once
3. **Supported formats**:
   - 📄 **PDF files** (.pdf) - Manuals, guides, documentation
   - 📝 **Word documents** (.docx) - Procedures, policies
   - 📃 **Text files** (.txt) - Configuration files, logs
   - 📋 **Markdown files** (.md) - Documentation, README files

## ✅ What happens after upload:
- Documents are processed and added to the knowledge base
- Content is split into searchable chunks
- Vector embeddings are created for semantic search
- Information becomes available for future questions

## 🎯 Best Practices:
- Upload official Trend Micro documentation
- Include product manuals and troubleshooting guides
- Add configuration examples and best practices
- Upload error logs for troubleshooting assistance

**Ready to upload? Click the 📎 attachment button below and select your files!**""",
        author="System"
    ).send()

@cl.action_callback("reset_conversations")
async def on_action_reset_conversations(action):
    """Handle reset conversations action"""
    try:
        import shutil
        import os

        # Ask for confirmation first
        await cl.Message(
            content="""🗑️ **Reset/Clear All Conversations**

⚠️ **Warning**: This action will permanently delete ALL conversation history!

**What will be deleted:**
- All previous conversations
- Chat history and timestamps
- Conversation metadata

**What will NOT be deleted:**
- Uploaded documents in knowledge base
- Vector database and embeddings
- Application settings

**To confirm deletion, type**: `/confirm_reset`
**To cancel, just continue chatting normally.**""",
            author="System"
        ).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ Error preparing reset: {str(e)}",
            author="System"
        ).send()

@cl.action_callback("view_history")
async def on_action_view_history(action):
    """Handle view history action"""
    conversations = assistant.list_conversations()

    if not conversations:
        await cl.Message(
            content="📭 **No Conversation History**\n\nYou haven't had any previous conversations yet. Start chatting to create your conversation history!",
            author="System"
        ).send()
        return

    history_content = "# 💬 Complete Conversation History\n\n"
    for i, conv in enumerate(conversations[:20], 1):  # Show last 20 conversations
        title = conv['title']
        history_content += f"**{i}. {title}**\n"
        history_content += f"   📅 {conv['last_updated'][:10]} | 💬 {conv['message_count']} messages\n"
        history_content += f"   🆔 ID: `{conv['id'][:8]}...`\n\n"

    history_content += f"\n**Total Conversations**: {len(conversations)}"

    if len(conversations) > 20:
        history_content += f"\n*Showing most recent 20 of {len(conversations)} conversations*"

    await cl.Message(
        content=history_content,
        author="System"
    ).send()

@cl.action_callback("new_conversation")
async def on_action_new_conversation(action):
    """Handle new conversation action"""
    # Start new conversation
    new_conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", new_conversation_id)
    cl.user_session.set("messages", [])

    await cl.Message(
        content="💬 **New Conversation Started**\n\nPrevious conversation has been saved. What would you like to know about Trend Micro products?",
        author="System"
    ).send()

@cl.action_callback("load_conversation")
async def on_action_load_conversation(action):
    """Handle load conversation action"""
    try:
        conversation_id = action.payload.get("conversation_id") if action.payload else action.value

        # Save current conversation before loading another
        current_conversation_id = cl.user_session.get("conversation_id")
        current_messages = cl.user_session.get("messages", [])
        if current_conversation_id and current_messages:
            assistant.save_conversation(current_conversation_id, current_messages)

        # Load the selected conversation
        conversation_data = assistant.load_conversation(conversation_id)

        if conversation_data:
            # Set the loaded conversation as current
            cl.user_session.set("conversation_id", conversation_id)
            cl.user_session.set("messages", conversation_data["messages"])

            # Display the conversation title and info
            await cl.Message(
                content=f"""📂 **Conversation Loaded**

**Title:** {conversation_data['title']}
**Created:** {conversation_data.get('created_at', 'Unknown')[:10]}
**Messages:** {conversation_data.get('message_count', 0)}

**Conversation History:**
""",
                author="System"
            ).send()

            # Display the conversation messages
            for msg in conversation_data["messages"]:
                if msg.get("role") == "user":
                    await cl.Message(
                        content=msg["content"],
                        author="User"
                    ).send()
                elif msg.get("role") == "assistant":
                    await cl.Message(
                        content=msg["content"],
                        author="Assistant"
                    ).send()

            await cl.Message(
                content="💬 **Conversation Loaded Successfully!**\n\nYou can continue this conversation or type `/reset` to start fresh.",
                author="System"
            ).send()

        else:
            await cl.Message(
                content="❌ **Error Loading Conversation**\n\nThe selected conversation could not be found or loaded.",
                author="System"
            ).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Error Loading Conversation**\n\nError: {str(e)}",
            author="System"
        ).send()

@cl.action_callback("view_stats")
async def on_action_view_stats(action):
    """Handle view stats action"""
    kb_stats = assistant.get_knowledge_base_stats()
    conversations = assistant.list_conversations()

    await cl.Message(
        content=f"""# 📊 Complete System Statistics

## 📚 Knowledge Base:
- **Total Documents**: {kb_stats['document_count']}
- **Searchable Chunks**: {kb_stats['chunk_count']}
- **Specialization**: {kb_stats['specialization']}
- **Database**: ChromaDB (Persistent)

## 💬 Conversation Data:
- **Total Conversations**: {len(conversations)}
- **Current Session**: {cl.user_session.get('conversation_id', 'New')[:8]}...
- **Session Messages**: {len(cl.user_session.get('messages', []))}

## 🔧 System Configuration:
- **AI Model**: {assistant.model}
- **Vector Database**: ChromaDB (Persistent)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **API Endpoint**: Trend Micro RDSec One AI
- **Status**: {'✅ Operational' if kb_stats['document_count'] >= 0 else '❌ Error'}

## 📈 Recent Activity:
{f'**Last Conversation**: {conversations[0]["last_updated"][:10]}' if conversations else '**No previous conversations**'}

Ready to answer questions about {assistant.specialization}!""",
        author="System"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_message = message.content.strip()
    conversation_id = cl.user_session.get("conversation_id")
    messages = cl.user_session.get("messages", [])

    # Handle file uploads
    if message.elements:
        await handle_file_upload(message.elements)
        return

    # Handle special commands
    if user_message.startswith('/'):
        await handle_command(user_message)
        return

    # Add user message to session
    messages.append({"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()})
    cl.user_session.set("messages", messages)

    # Show typing indicator with RAG search
    async with cl.Step(name="🔍 Searching Knowledge Base", type="tool") as step:
        step.output = "Searching through uploaded documents..."

        # Get max docs from sidebar settings
        settings = cl.user_session.get("settings", {})
        max_docs = settings.get("max_docs", 5)

        # Search knowledge base
        relevant_docs = assistant.search_knowledge_base(user_message, max_results=max_docs)
        if relevant_docs:
            sources = [doc['metadata'].get('filename', 'Unknown') for doc in relevant_docs]
            step.output = f"Found relevant information in: {', '.join(set(sources))}"
        else:
            step.output = "No specific documents found, using general knowledge..."

    async with cl.Step(name="🤔 Generating Response", type="tool") as step:
        step.output = "Generating response with AI..."

        # Debug: Print conversation context
        print(f"DEBUG: Current messages count: {len(messages)}")
        if messages:
            print(f"DEBUG: Last few messages:")
            for i, msg in enumerate(messages[-3:], 1):
                print(f"  {i}. {msg.get('role', 'unknown')}: {msg.get('content', '')[:50]}...")

        # Get response from RAG assistant with conversation context
        response = await assistant.get_response(user_message, messages)

        step.output = "✅ Response generated!"

    # Add assistant response to session
    messages.append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})
    cl.user_session.set("messages", messages)

    # Save conversation
    assistant.save_conversation(conversation_id, messages)

    # Add source information if enabled in sidebar
    settings = cl.user_session.get("settings", {})
    show_sources = settings.get("show_sources", True)

    if show_sources and relevant_docs:
        sources_text = "\n\n**📚 Sources Used:**\n"
        for i, doc in enumerate(relevant_docs[:3], 1):  # Show top 3 sources
            filename = doc['metadata'].get('filename', 'Unknown')
            similarity = doc['similarity']
            sources_text += f"{i}. {filename} (Relevance: {similarity:.0%})\n"

        response_with_sources = response + sources_text
    else:
        response_with_sources = response

    # Send response
    await cl.Message(content=response_with_sources, author="Assistant").send()

async def handle_file_upload(elements):
    """Handle document uploads"""
    try:
        for element in elements:
            if hasattr(element, 'content') and hasattr(element, 'name'):
                # Determine file type and process accordingly
                filename = element.name.lower()

                if filename.endswith('.pdf'):
                    content = assistant.process_pdf(element)
                elif filename.endswith('.docx'):
                    content = assistant.process_docx(element)
                elif filename.endswith(('.txt', '.md')):
                    content = assistant.process_text_file(element)
                else:
                    await cl.Message(
                        content=f"❌ Unsupported file type: {filename}\n\nSupported types: PDF, DOCX, TXT, MD",
                        author="System"
                    ).send()
                    continue

                # Add to knowledge base
                if assistant.add_document_to_kb(element.name, content):
                    kb_stats = assistant.get_knowledge_base_stats()
                    await cl.Message(
                        content=f"""✅ **Document Added Successfully!**

**File**: {element.name}
**Size**: {len(content):,} characters
**Status**: Added to knowledge base

📊 **Updated Knowledge Base**:
- **Total Documents**: {kb_stats['document_count']}
- **Total Chunks**: {kb_stats['chunk_count']}

Your document is now searchable and will be used to answer relevant questions!""",
                        author="System"
                    ).send()
                else:
                    await cl.Message(
                        content=f"⚠️ **Document Already Exists**: {element.name}\n\nThis document (or identical content) is already in the knowledge base.",
                        author="System"
                    ).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Error Processing Upload**: {str(e)}",
            author="System"
        ).send()

async def handle_command(command: str):
    """Handle special commands"""
    if command == "/reset" or command == "/new":
        # Start new conversation
        new_conversation_id = str(uuid.uuid4())
        cl.user_session.set("conversation_id", new_conversation_id)
        cl.user_session.set("messages", [])

        await cl.Message(
            content="🔄 **New Conversation Started**\n\nPrevious conversation saved. How can I help you with Trend Micro products?",
            author="System"
        ).send()

    elif command == "/help":
        await cl.Message(
            content=f"""# 🆘 Trend Micro Knowledge Assistant - Help

## 📋 Available Commands:
- `/new` or `/reset` - Start a new conversation
- `/upload` - Instructions for uploading documents
- `/fetch [topic]` - Fetch content from Help Center (e.g., `/fetch installation`)
- `/history` - Show conversation history
- `/stats` - Show knowledge base statistics
- `/help` - Show this help message

## 🎯 What I specialize in:
- **Product Support**: {assistant.specialization} product features and capabilities
- **Installation**: Setup and configuration guides
- **Troubleshooting**: Technical issues and solutions
- **Best Practices**: Recommendations and optimization
- **Licensing**: Activation and license management

## 📁 Document Upload:
Click the attachment button (📎) to upload:
- **PDF files**: Product manuals, guides, documentation
- **Word documents**: Procedures, policies, instructions
- **Text files**: Configuration files, logs, notes
- **Markdown files**: Documentation, README files

## 🔍 How RAG Works:
1. Upload documents to build knowledge base
2. Ask questions about your uploaded content
3. I'll search and use relevant information to answer
4. All conversations and documents are saved permanently

## 🚀 Tips:
- Be specific about your Trend Micro product
- Include error messages if you're troubleshooting
- Upload relevant documentation for better answers
- Use `/new` to start fresh conversations

**Just ask me anything about {assistant.specialization}!**""",
            author="System"
        ).send()

    elif command == "/upload":
        await cl.Message(
            content="""# 📁 Document Upload Instructions

## 🔄 How to Upload Documents:

1. **Click the attachment button** (📎) in the message input area
2. **Select your files** - supports multiple files at once
3. **Supported formats**:
   - 📄 **PDF files** (.pdf) - Manuals, guides, documentation
   - 📝 **Word documents** (.docx) - Procedures, policies
   - 📃 **Text files** (.txt) - Configuration files, logs
   - 📋 **Markdown files** (.md) - Documentation, README files

## ✅ What happens after upload:
- Documents are processed and added to the knowledge base
- Content is split into searchable chunks
- Vector embeddings are created for semantic search
- Information becomes available for future questions

## 🎯 Best Practices:
- Upload official Trend Micro documentation
- Include product manuals and troubleshooting guides
- Add configuration examples and best practices
- Upload error logs for troubleshooting assistance

## 💾 Persistence:
- All uploaded documents are saved permanently
- Knowledge base persists between sessions
- Documents remain searchable across conversations

**Ready to upload? Click the 📎 attachment button and select your files!**""",
            author="System"
        ).send()

    elif command == "/stats":
        kb_stats = assistant.get_knowledge_base_stats()
        conversations = assistant.list_conversations()

        await cl.Message(
            content=f"""# 📊 Knowledge Base Statistics

## 📚 Document Storage:
- **Total Documents**: {kb_stats['document_count']}
- **Searchable Chunks**: {kb_stats['chunk_count']}
- **Specialization**: {kb_stats['specialization']}

## 💬 Conversation History:
- **Total Conversations**: {len(conversations)}
- **Current Session**: {cl.user_session.get('conversation_id', 'New')[:8]}...

## 🔧 System Information:
- **AI Model**: {assistant.model}
- **Vector Database**: ChromaDB (Persistent)
- **Embeddings**: SentenceTransformers
- **Status**: {'✅ Operational' if kb_stats['document_count'] >= 0 else '❌ Error'}

## 📈 Recent Activity:
{f'**Last Conversation**: {conversations[0]["last_updated"][:10]}' if conversations else '**No previous conversations**'}

Ready to answer questions about {assistant.specialization}!""",
            author="System"
        ).send()

    elif command == "/history":
        try:
            # Save current conversation first
            current_conversation_id = cl.user_session.get("conversation_id")
            current_messages = cl.user_session.get("messages", [])

            if current_conversation_id and current_messages:
                assistant.save_conversation(current_conversation_id, current_messages)

            conversations = assistant.list_conversations()

            if not conversations:
                await cl.Message(
                    content="📭 **No Conversation History**\n\nYou haven't had any previous conversations yet. Start chatting to create your conversation history!\n\n💡 **Tip**: After chatting, your conversations will appear here for easy access.",
                    author="System"
                ).send()
                return

            # Show conversation history with direct load commands
            history_content = f"""💬 **Conversation History** ({len(conversations)} conversations)

## 📋 **Available Conversations:**

"""

            for i, conv in enumerate(conversations[:15], 1):  # Show last 15 conversations
                # Safe extraction with null checks
                title = conv.get('title', 'Untitled Conversation') or 'Untitled Conversation'
                last_updated = conv.get('last_updated', 'Unknown') or 'Unknown'
                date = last_updated[:10] if last_updated != 'Unknown' else 'Unknown'
                msg_count = conv.get('message_count', 0) or 0
                conv_id_full = conv.get('id', '') or ''
                conv_id = conv_id_full[:8] if conv_id_full else 'unknown'

                # Truncate title if too long
                display_title = title[:40] + "..." if len(title) > 40 else title

                # Clean title for load command (remove special characters)
                clean_title = ''.join(c.lower() if c.isalnum() else '_' for c in title[:20])

                history_content += f"""**{i}. {display_title}**
📅 {date} | 💬 {msg_count} messages | 🆔 {conv_id}...
**Load:** `/load {i}` or `/load {clean_title}` or `/load {conv_id}`

"""

            history_content += """---

## 💡 **How to Load Conversations:**
- **By Number**: `/load 1` or `/load 2` (loads 1st, 2nd conversation from list above)
- **By Title**: `/load trend_micro` (matches partial title)
- **By ID**: `/load 12345678` (using the 8-character ID shown above)

**Example**: Type `/load 1` to load the first conversation in the list above."""

            await cl.Message(
                content=history_content,
                author="System"
            ).send()

        except Exception as e:
            await cl.Message(
                content=f"❌ **Error Loading History**\n\nError: {str(e)}",
                author="System"
            ).send()

    elif command == "/confirm_reset":
        # Actually delete all conversations
        try:
            import shutil
            conversations_dir = assistant.conversations_dir

            # Remove all conversation files
            if conversations_dir.exists():
                shutil.rmtree(conversations_dir)
                conversations_dir.mkdir(exist_ok=True)

            # Reset current session
            new_conversation_id = str(uuid.uuid4())
            cl.user_session.set("conversation_id", new_conversation_id)
            cl.user_session.set("messages", [])

            await cl.Message(
                content="""✅ **All Conversations Deleted Successfully**

All conversation history has been permanently removed from the system.

**What was deleted:**
- All previous conversation files
- Chat history and timestamps
- Conversation metadata

**What remains intact:**
- Knowledge base documents
- Vector database and embeddings
- System configuration

**You now have a fresh start!** How can I help you with Trend Micro products?""",
                author="System"
            ).send()

        except Exception as e:
            await cl.Message(
                content=f"❌ **Error during reset**: {str(e)}\n\nPlease try again or contact your administrator.",
                author="System"
            ).send()

    elif command.startswith("/load "):
        # Handle load conversation command
        try:
            # Extract argument from command
            parts = command.split(" ", 1)
            if len(parts) < 2:
                await cl.Message(
                    content="""❓ **Invalid Load Command**

**Usage Options:**
- `/load 1` - Load 1st conversation from `/history` list
- `/load 2` - Load 2nd conversation from `/history` list
- `/load trend_micro` - Load by partial title match
- `/load 12345678` - Load by 8-character ID
- `/load selected` - Load from sidebar dropdown (if available)

💡 **Tip:** Use `/history` to see numbered list of conversations first!""",
                    author="System"
                ).send()
                return

            search_term = parts[1].strip()
            conversations = assistant.list_conversations()
            target_conversation = None

            # Check if it's a number (1, 2, 3, etc.) for direct list access
            if search_term.isdigit():
                conv_index = int(search_term) - 1  # Convert to 0-based index
                if 0 <= conv_index < len(conversations):
                    target_conversation = conversations[conv_index]
                else:
                    await cl.Message(
                        content=f"❌ **Invalid Number**\n\nConversation number `{search_term}` not found.\n\n💡 **Try:** `/history` to see available conversations numbered 1-{len(conversations)}",
                        author="System"
                    ).send()
                    return
            elif search_term.lower() == "selected":
                # Get the selected conversation from sidebar
                settings = cl.user_session.get("settings", {})
                conversation_mapping = cl.user_session.get("conversation_mapping", {})

                if "conversation_selector" in settings and conversation_mapping:
                    selected_display = settings["conversation_selector"]
                    conversation_id = conversation_mapping.get(selected_display)

                    if conversation_id:
                        for conv in conversations:
                            if conv.get('id') == conversation_id:
                                target_conversation = conv
                                break

                    if not target_conversation:
                        await cl.Message(
                            content="❌ **No Conversation Selected**\n\nPlease select a conversation from the sidebar dropdown first, then use `/load selected`.",
                            author="System"
                        ).send()
                        return
                else:
                    await cl.Message(
                        content="❌ **No Conversation Selected**\n\nPlease select a conversation from the sidebar dropdown first, or use `/load [number]` from `/history` list.",
                        author="System"
                    ).send()
                    return
            else:
                # Search by title or partial ID
                for conv in conversations:
                    title = conv.get('title', '').lower()
                    conv_id = conv.get('id', '')

                    # Match by title (partial) or ID (partial)
                    if (search_term.lower() in title or
                        conv_id.startswith(search_term) or
                        title.startswith(search_term.lower())):
                        target_conversation = conv
                        break

                if not target_conversation:
                    await cl.Message(
                        content=f"❌ **Conversation Not Found**\n\nNo conversation found matching `{search_term}`\n\n💡 **Try:**\n- `/history` to see numbered list\n- `/load 1` or `/load 2` (by number)\n- `/load [part of title]` (by title)\n- `/load [8-char-id]` (by ID)",
                        author="System"
                    ).send()
                    return

            # Save current conversation before loading another
            current_conversation_id = cl.user_session.get("conversation_id")
            current_messages = cl.user_session.get("messages", [])
            if current_conversation_id and current_messages:
                assistant.save_conversation(current_conversation_id, current_messages)

            # Load the selected conversation
            conversation_data = assistant.load_conversation(target_conversation['id'])

            if conversation_data:
                # Set the loaded conversation as current
                cl.user_session.set("conversation_id", target_conversation['id'])
                cl.user_session.set("messages", conversation_data["messages"])

                # Display the conversation title and info
                await cl.Message(
                    content=f"""📂 **Loading Conversation**

**Title:** {conversation_data['title']}
**Created:** {conversation_data.get('created_at', 'Unknown')[:10]}
**Messages:** {conversation_data.get('message_count', 0)}
**ID:** {target_conversation['id'][:8]}...

**Previous Conversation History:**""",
                    author="System"
                ).send()

                # Display ALL the conversation messages to show the full history
                messages = conversation_data.get("messages", [])
                if messages:
                    for i, msg in enumerate(messages, 1):
                        if msg.get("role") == "user":
                            await cl.Message(
                                content=f"**[{i}] User:** {msg['content']}",
                                author="📋 History"
                            ).send()
                        elif msg.get("role") == "assistant":
                            await cl.Message(
                                content=f"**[{i}] Assistant:** {msg['content']}",
                                author="📋 History"
                            ).send()
                else:
                    await cl.Message(
                        content="No messages found in this conversation.",
                        author="System"
                    ).send()

                await cl.Message(
                    content=f"""✅ **Conversation Loaded Successfully!**

**Summary:**
- **{len(messages)}** messages displayed above
- You can now continue this conversation where you left off
- Type `/reset` to start a new conversation instead
- Type `/history` to see other available conversations

**What would you like to ask next?**""",
                    author="System"
                ).send()

            else:
                await cl.Message(
                    content="❌ **Error Loading Conversation**\n\nThe conversation file could not be loaded.",
                    author="System"
                ).send()

        except Exception as e:
            await cl.Message(
                content=f"❌ **Error Loading Conversation**\n\nError: {str(e)}",
                author="System"
            ).send()

    elif command.startswith("/fetch "):
        # Handle fetch Help Center content command
        try:
            # Extract search term from command
            parts = command.split(" ", 1)
            if len(parts) < 2:
                await cl.Message(
                    content="""❓ **Invalid Fetch Command**

**Usage:** `/fetch [topic]`

**Examples:**
- `/fetch installation` - Get installation guides
- `/fetch troubleshooting` - Get troubleshooting help
- `/fetch licensing` - Get licensing information
- `/fetch virus scan` - Get scanning help
- `/fetch error messages` - Get error solutions

💡 **Tip:** Be specific with your topic for better results!""",
                    author="System"
                ).send()
                return

            search_topic = parts[1].strip()

            await cl.Message(
                content=f"🔍 **Fetching Help Center Content**\n\nSearching Trend Micro Help Center for: **{search_topic}**\n\nThis may take a few moments...",
                author="System"
            ).send()

            # Fetch content from Help Center
            added_count = assistant.add_help_center_to_kb(search_topic)

            if added_count > 0:
                kb_stats = assistant.get_knowledge_base_stats()
                await cl.Message(
                    content=f"""✅ **Help Center Content Added Successfully!**

**Search Topic**: {search_topic}
**Articles Added**: {added_count} new articles
**Status**: Content now available in knowledge base

📊 **Updated Knowledge Base**:
- **Total Documents**: {kb_stats['document_count']}
- **Total Chunks**: {kb_stats['chunk_count']}

**What's Next:**
- Ask questions about **{search_topic}**
- The assistant now has fresh information from the Help Center
- Content will be automatically used to answer relevant questions

💡 **Example**: Try asking "How do I {search_topic}?" or "Tell me about {search_topic}"!""",
                    author="System"
                ).send()
            else:
                await cl.Message(
                    content=f"""⚠️ **No New Content Added**

**Search Topic**: {search_topic}

**Possible Reasons:**
- Content may already exist in the knowledge base
- No relevant articles found for this topic
- Help Center may be temporarily unavailable

**What to Try:**
- Use a different search term (e.g., "installation", "troubleshooting")
- Try more specific keywords
- Check if the content already exists with regular questions

**Still Available:** You can still ask questions - I'll search both local knowledge and Help Center in real-time!""",
                    author="System"
                ).send()

        except Exception as e:
            await cl.Message(
                content=f"❌ **Error Fetching Help Center Content**\n\nError: {str(e)}\n\n**Please try again or contact your administrator.**",
                author="System"
            ).send()

    else:
        await cl.Message(
            content=f"❓ **Unknown Command**: `{command}`\n\nUse `/help` to see available commands.",
            author="System"
        ).send()

@cl.on_stop
async def stop():
    """Save conversation when session ends"""
    try:
        conversation_id = cl.user_session.get("conversation_id")
        messages = cl.user_session.get("messages", [])

        if conversation_id and messages:
            assistant.save_conversation(conversation_id, messages)
            print(f"Conversation {conversation_id} saved with {len(messages)} messages")

    except Exception as e:
        print(f"Error saving conversation on stop: {e}")

if __name__ == "__main__":
    # This will be called when running with: chainlit run complete_rag_app.py
    pass