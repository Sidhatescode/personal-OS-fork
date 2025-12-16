import os.path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
import json
from openai import OpenAI
import re
from datetime import datetime, timedelta
import time

# Load secrets from secrets.json
def load_secrets():
    try:
        with open('secrets.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("secrets.json file not found. Please create it with your OpenAI API key.")
    except json.JSONDecodeError:
        raise ValueError("secrets.json is not properly formatted JSON.")

# Initialize OpenAI client with API key from secrets
secrets = load_secrets()
client = OpenAI(api_key=secrets['openai_api_key'])

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/gmail.send']

# Configuration
NOTE_EMAIL = "haldarsiddharth+notes@gmail.com"  # CHANGE THIS to your email address
NOTES_FILE = "processed_notes.json"
PROCESSED_LABEL = "PROCESSED_NOTES"  # Gmail label to mark processed notes
RESULT_EMAIL = "haldarsiddharth@gmail.com"  # Email address to send processed notes to

def load_prompt_template():
    """Loads the prompt template from prompt.txt"""
    try:
        with open('prompt.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError("prompt.txt file not found. Please create it with your note processing instructions.")

def process_note_with_ai(note_content, note_date):
    """Processes note using GPT-4o to split, categorize, and tag."""
    try:
        # Load the prompt template
        prompt_template = load_prompt_template()
        
        # Format the prompt with the note details
        prompt = prompt_template.format(
            note_content=note_content,
            note_date=note_date
        )
        
        # Show note length for debugging
        note_length = len(note_content)
        print(f"   📏 Note length: {note_length} characters")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a personal knowledge management assistant that preserves exact wording while organizing thoughts."},
                {"role": "user", "content": prompt}
            ],
            timeout=180.0  # 180 second timeout (3 minutes)
        )
        result = json.loads(response.choices[0].message.content)
        
        # Validate the response structure
        if 'atomic_notes' not in result:
            return {"error": "AI response missing 'atomic_notes' field"}
        
        return result
    except KeyError as e:
        print(f"Debug - KeyError: Missing key {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"KeyError: {str(e)}"}
    except json.JSONDecodeError as e:
        print(f"Debug - JSON decode error: {str(e)}")
        return {"error": f"Failed to parse AI response as JSON: {str(e)}"}
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            print(f"⏱️  Request timed out after 180 seconds. Note may be too long ({len(note_content)} chars).")
            return {"error": f"Request timed out. Note length: {len(note_content)} characters. Try splitting into smaller notes."}
        print(f"Debug - Error details: {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to process: {error_msg}"}

def get_gmail_service():
    """Gets or creates Gmail API service."""
    creds = None
    
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def clean_email_content(text):
    """Cleans email content while preserving the actual note content."""
    
    # Remove style and script tags and their contents
    text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text)
    text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text)
    
    # Remove image tags but keep alt text
    text = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r'\1', text)
    text = re.sub(r'<img[^>]*>', '', text)
    
    # Remove common email footer patterns
    patterns_to_remove = [
        r'Copyright ©.*?(?=\n|$)',
        r'You are receiving this email because.*?(?=\n|$)',
        r'To connect with us.*?(?=\n|$)',
        r'Our mailing address.*?(?=\n|$)',
        r'Unsubscribe.*?(?=\n|$)',
        r'Add .* to your address book.*?(?=\n|$)',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove HTML attributes
    text = re.sub(r'style="[^"]*"', '', text)
    text = re.sub(r'class="[^"]*"', '', text)
    text = re.sub(r'width="[^"]*"', '', text)
    text = re.sub(r'height="[^"]*"', '', text)
    
    # Remove URLs but preserve link text
    text = re.sub(r'<a[^>]*href="[^"]*"[^>]*>([^<]+)</a>', r'\1', text)
    text = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/]+={0,2}', '', text)
    
    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def create_label_if_needed(service, label_name):
    """Creates a Gmail label if it doesn't exist."""
    results = service.users().labels().list(userId='me').execute()
    existing_labels = {label['name']: label['id'] for label in results.get('labels', [])}
    
    if label_name not in existing_labels:
        label_body = {
            'name': label_name,
            'messageListVisibility': 'show',
            'labelListVisibility': 'labelShow'
        }
        created_label = service.users().labels().create(userId='me', body=label_body).execute()
        return created_label['id']
    
    return existing_labels[label_name]

def mark_email_as_processed(service, message_id, label_id):
    """Marks an email as processed by adding a label and marking as read."""
    try:
        service.users().messages().modify(
            userId='me',
            id=message_id,
            body={
                'addLabelIds': [label_id],
                'removeLabelIds': ['UNREAD']
            }
        ).execute()
    except Exception as e:
        print(f"Error marking email as processed: {str(e)}")

def format_notes_for_email(atomic_notes):
    """Formats atomic notes into a readable email body."""
    if not atomic_notes:
        return "No notes were processed."
    
    email_body = f"📝 Processed {len(atomic_notes)} atomic note(s):\n\n"
    email_body += "=" * 60 + "\n\n"
    
    for i, note in enumerate(atomic_notes, 1):
        email_body += f"[{i}] {note['original_text']}\n\n"
        email_body += f"📁 Categories: {', '.join(note['categories'])}\n"
        email_body += f"🏷️  Tags: {', '.join(note['tags'])}\n"
        if note.get('context'):
            email_body += f"💡 Context: {note['context']}\n"
        email_body += f"📅 Source: {note.get('source_subject', 'N/A')} ({note.get('source_date', 'N/A')})\n"
        email_body += "\n" + "-" * 60 + "\n\n"
    
    return email_body

def send_processed_notes_email(service, atomic_notes, original_subject):
    """Sends an email with the processed atomic notes."""
    try:
        # Format the notes
        email_body = format_notes_for_email(atomic_notes)
        
        # Create the email message
        message = MIMEText(email_body)
        message['to'] = RESULT_EMAIL
        message['subject'] = f"Processed Notes: {original_subject}"
        
        # Encode the message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # Send the email
        send_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        
        print(f"📧 Email sent successfully! Message ID: {send_message['id']}")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return False

def load_existing_notes():
    """Load previously processed notes from file."""
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, 'r') as f:
            return json.load(f)
    return {"notes": []}

def save_notes(notes_data):
    """Save notes to file."""
    with open(NOTES_FILE, 'w') as f:
        json.dump(notes_data, f, indent=2)

def get_email_content(service, message_id):
    """Extract content from an email."""
    msg = service.users().messages().get(userId='me', id=message_id, format='full').execute()
    
    headers = msg['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'No Sender')
    date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'No Date')
    
    # Better email body extraction for multipart emails
    def extract_body(payload):
        body = ''
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part.get('body', {}):
                        data = part['body']['data']
                        body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                elif part['mimeType'] == 'text/html' and not body:
                    # Fallback to HTML if no plain text found
                    if 'data' in part.get('body', {}):
                        data = part['body']['data']
                        body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                elif 'parts' in part:
                    # Recursive for nested parts
                    nested_body = extract_body(part)
                    if nested_body:
                        body += nested_body
        else:
            # Single part email
            if payload.get('mimeType') == 'text/plain' or payload.get('mimeType') == 'text/html':
                if 'data' in payload.get('body', {}):
                    data = payload['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        
        return body
    
    content = extract_body(msg['payload'])
    if content:
        content = clean_email_content(content)
    else:
        content = 'No content'
    
    return {
        'subject': subject,
        'sender': sender,
        'date': date,
        'content': content,
        'message_id': message_id
    }

def process_notes_from_email():
    """Main function to process notes sent via email."""
    service = get_gmail_service()
    
    # Create the processed label if needed
    processed_label_id = create_label_if_needed(service, PROCESSED_LABEL)
    
    # Search for unread emails to the notes address that haven't been processed
    query = f'to:{NOTE_EMAIL} is:unread -label:{PROCESSED_LABEL}'
    
    try:
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        
        if not messages:
            print('No new notes found.')
            return
        
        print(f'Found {len(messages)} new note(s) to process...')
        print('=' * 60)
        
        # Load existing notes
        all_notes = load_existing_notes()
        
        for i, message in enumerate(messages, 1):
            email_data = get_email_content(service, message['id'])
            
            print(f"\n[{i}/{len(messages)}] Processing note:")
            print(f"Subject: {email_data['subject']}")
            print(f"From: {email_data['sender']}")
            content_length = len(email_data['content'])
            print(f"Content preview: {email_data['content'][:100]}...")
            if content_length > 10000:
                print(f"⚠️  Warning: Note is very long ({content_length} characters). This may take longer to process.")
            print("🤖 Sending to AI for processing...")
            
            # Process with AI
            ai_result = process_note_with_ai(email_data['content'], email_data['date'])
            
            if 'error' in ai_result:
                print(f"❌ Error: {ai_result['error']}")
                continue
            
            if ai_result and 'atomic_notes' in ai_result:
                # Add metadata to each atomic note
                for atomic_note in ai_result['atomic_notes']:
                    atomic_note['source_subject'] = email_data['subject']
                    atomic_note['source_sender'] = email_data['sender']
                    atomic_note['source_date'] = email_data['date']
                    atomic_note['processed_at'] = datetime.now().isoformat()
                    atomic_note['message_id'] = email_data['message_id']
                
                # Add to our collection
                all_notes['notes'].extend(ai_result['atomic_notes'])
                
                print(f"✓ Split into {len(ai_result['atomic_notes'])} atomic note(s)")
                
                # Show each atomic note
                for j, note in enumerate(ai_result['atomic_notes'], 1):
                    print(f"\n  [{j}] \"{note['original_text']}\"")
                    print(f"      📁 Categories: {', '.join(note['categories'])}")
                    print(f"      🏷️  Tags: {', '.join(note['tags'])}")
                    if note.get('context'):
                        print(f"      💡 Context: {note['context']}")
                    print()
                
                # Send email with processed notes
                print("📧 Sending processed notes via email...")
                send_processed_notes_email(service, ai_result['atomic_notes'], email_data['subject'])
            
            # Mark email as processed
            mark_email_as_processed(service, message['id'], processed_label_id)
            
            # Rate limiting
            if i < len(messages):
                time.sleep(2)
        
        # Save all notes
        save_notes(all_notes)
        print("\n" + "=" * 60)
        print(f"✅ Processing complete!")
        print(f"📊 Total notes in system: {len(all_notes['notes'])}")
        print(f"💾 Saved to: {NOTES_FILE}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def search_notes(query):
    """Simple search function to find notes."""
    all_notes = load_existing_notes()
    
    if not all_notes['notes']:
        print("No notes in system yet. Process some notes first!")
        return
    
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 60)
    
    matches = []
    query_lower = query.lower()
    
    for note in all_notes['notes']:
        # Search in text, categories, tags, and context
        if (query_lower in note['original_text'].lower() or
            any(query_lower in cat.lower() for cat in note['categories']) or
            any(query_lower in tag.lower() for tag in note['tags']) or
            query_lower in note.get('context', '').lower()):
            matches.append(note)
    
    if not matches:
        print("No matches found.")
        return
    
    print(f"Found {len(matches)} matching note(s):\n")
    
    for i, note in enumerate(matches, 1):
        print(f"{i}. \"{note['original_text']}\"")
        print(f"   📁 Categories: {', '.join(note['categories'])}")
        print(f"   🏷️  Tags: {', '.join(note['tags'])}")
        print(f"   📅 From: {note['source_subject']} ({note['source_date']})")
        print()

def browse_by_category(category):
    """Browse all notes in a specific category."""
    all_notes = load_existing_notes()
    
    if not all_notes['notes']:
        print("No notes in system yet.")
        return
    
    matches = [n for n in all_notes['notes'] 
               if any(category.lower() in cat.lower() for cat in n['categories'])]
    
    print(f"\n📂 {len(matches)} note(s) in category '{category}':")
    print("=" * 60)
    
    for i, note in enumerate(matches, 1):
        print(f"{i}. \"{note['original_text']}\"")
        print(f"   🏷️  Tags: {', '.join(note['tags'])}")
        print()

def show_stats():
    """Show statistics about your notes."""
    all_notes = load_existing_notes()
    
    if not all_notes['notes']:
        print("No notes in system yet.")
        return
    
    # Count by category
    category_counts = {}
    all_tags = []
    
    for note in all_notes['notes']:
        for cat in note['categories']:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        all_tags.extend(note['tags'])
    
    print("\n📊 Your Knowledge Base Statistics")
    print("=" * 60)
    print(f"Total notes: {len(all_notes['notes'])}")
    print(f"\nNotes by category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {cat}: {count}")
    
    # Most common tags
    from collections import Counter
    top_tags = Counter(all_tags).most_common(10)
    print(f"\nTop 10 tags:")
    for tag, count in top_tags:
        print(f"  • {tag}: {count}")

if __name__ == '__main__':
    # Process new notes from email
    process_notes_from_email()
    
    # Uncomment to search or browse:
    # search_notes("copywriting")
    # browse_by_category("Swipe File")
    # show_stats()