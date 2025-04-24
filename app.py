from flask import Flask, request, jsonify, session
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import traceback
import uuid
from datetime import datetime
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
vectorstore = PineconeVectorStore(
    index_name="insurance",
    embedding=embeddings,
    namespace="ns1",
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
)

conversations = {}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query']
        
        session_id = data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            conversations[session_id] = {
                'history': [],
                'escalated': False,
                'escalation_reason': None,
                'created_at': datetime.now().isoformat()
            }
        elif session_id not in conversations:
            conversations[session_id] = {
                'history': [],
                'escalated': False,
                'escalation_reason': None,
                'created_at': datetime.now().isoformat()
            }
        
        # Check if already escalated
        if conversations[session_id]['escalated']:
            conversations[session_id]['history'].append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({
                'session_id': session_id,
                'response': "Your conversation has been escalated to a human agent. They will respond shortly.",
                'escalated': True,
                'escalation_reason': conversations[session_id]['escalation_reason']
            })
        
        # Process query with vector search
        results_with_score = vectorstore.similarity_search_with_score(query, k=2)
        
        # Check confidence of best result
        confidence_threshold = 0.75  # Adjust as needed
        best_score = results_with_score[0][1] if results_with_score else 0
        
        # Determine if we need to escalate
        needs_escalation = False
        escalation_reason = None
        
        # Case 1: Low confidence score
        if best_score < confidence_threshold:
            needs_escalation = True
            escalation_reason = "The AI couldn't find a sufficiently relevant answer in our knowledge base."
        
        # Case 2: Contains specific keywords for escalation
        escalation_keywords = ["speak to agent", "talk to human", "speak to representative", "talk to person", "human agent"]
        if any(keyword in query.lower() for keyword in escalation_keywords):
            needs_escalation = True
            escalation_reason = "You requested to speak with a human agent."
        
        # Case 3: Complex queries that might need human assistance
        complex_patterns = [
            r"dispute.*claim",
            r"appeal.*denial",
            r"lawsuit",
            r"legal.*action",
            r"cancel.*policy",
            r"refund.*premium"
        ]
        if any(re.search(pattern, query.lower()) for pattern in complex_patterns):
            needs_escalation = True
            escalation_reason = "Your request involves a complex insurance matter best handled by a specialist."
        
        # Add message to history
        conversations[session_id]['history'].append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        })
        
        if needs_escalation:
            conversations[session_id]['escalated'] = True
            conversations[session_id]['escalation_reason'] = escalation_reason
            
            
            response_message = f"I'll connect you with a human agent who can better assist you. {escalation_reason}"
            
            conversations[session_id]['history'].append({
                'role': 'assistant',
                'content': response_message,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({
                'session_id': session_id,
                'response': response_message,
                'escalated': True,
                'escalation_reason': escalation_reason
            })
        
        formatted_results = []
        for i, (doc, score) in enumerate(results_with_score):
            formatted_results.append({
                'index': i + 1,
                'score': float(score),
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        response_message = "Based on our insurance policies:\n\n"
        if formatted_results:
            response_message += formatted_results[0]['content']
        else:
            response_message = "I don't have specific information about that in our knowledge base. Would you like to speak with a human agent for more assistance?"
        
        conversations[session_id]['history'].append({
            'role': 'assistant',
            'content': response_message,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'session_id': session_id,
            'response': response_message,
            'escalated': False,
            'results': formatted_results
        })
    
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        # Escalate on error
        session_id = request.get_json().get('session_id', str(uuid.uuid4()))
        if session_id in conversations:
            conversations[session_id]['escalated'] = True
            conversations[session_id]['escalation_reason'] = "An error occurred while processing your request."
        
        return jsonify({
            'error': str(e),
            'session_id': session_id,
            'escalated': True,
            'escalation_reason': "An error occurred while processing your request."
        }), 500

@app.route('/agent/respond', methods=['POST'])
def agent_respond():
    try:
        data = request.get_json()
        if not data or 'session_id' not in data or 'response' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        session_id = data['session_id']
        response = data['response']
        
        if session_id not in conversations:
            return jsonify({'error': 'Conversation not found'}), 404
        
        if not conversations[session_id]['escalated']:
            return jsonify({'error': 'This conversation has not been escalated'}), 400
        
        # Add agent response to history
        conversations[session_id]['history'].append({
            'role': 'agent',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'session_id': session_id,
            'status': 'success',
            'message': 'Response added to conversation'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

# endpoint to get conversation history
@app.route('/conversation/<session_id>', methods=['GET'])
def get_conversation(session_id):
    if session_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404
    
    return jsonify({
        'session_id': session_id,
        'history': conversations[session_id]['history'],
        'escalated': conversations[session_id]['escalated'],
        'escalation_reason': conversations[session_id]['escalation_reason']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)