import pandas as pd
import openai
from datetime import datetime, timezone
import smtplib
import imaplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import email
import json
import os
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import numpy as np
import time
import notion_client

# --- Configuration & Initialization ---

CONFIG_FILE = "config.json"

def load_secrets_from_env():
    """Laadt geheimen uit environment variables op de server."""
    print("   -> Geheimen laden uit environment variables...")
    try:
        # --- FIXED: Reconstruct the google_credentials dict from individual env vars ---
        google_creds = {
            "type": os.environ.get("GOOGLE_TYPE"),
            "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
            "private_key_id": os.environ.get("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("GOOGLE_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.environ.get("GOOGLE_CLIENT_EMAIL"),
            "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
            "auth_uri": os.environ.get("GOOGLE_AUTH_URI"),
            "token_uri": os.environ.get("GOOGLE_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.environ.get("GOOGLE_AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.environ.get("GOOGLE_CLIENT_X509_CERT_URL")
        }

        secrets = {
            "openai": {
                "api_key": os.environ.get("OPENAI_API_KEY")
            },
            "email": {
                "password": os.environ.get("EMAIL_PASSWORD")
            },
            "notion": {
                "api_key_notion": os.environ.get("NOTION_API_KEY"),
                "database_id": os.environ.get("NOTION_DATABASE_ID")
            },
            "google_credentials": google_creds
        }
        
        # Controleer of alle geheimen zijn geladen
        if not all([
            secrets["openai"]["api_key"],
            secrets["email"]["password"],
            secrets["notion"]["api_key_notion"],
            secrets["notion"]["database_id"],
            secrets["google_credentials"]["private_key"] # Check a key field
        ]):
            raise ValueError("Een of meer environment variables zijn niet ingesteld.")
            
        print("✅ Geheimen succesvol geladen.")
        return secrets
    except (ValueError, TypeError) as e:
        print(f"❌ Fout bij het laden van de geheimen uit environment variables: {e}")
        print("   Controleer of alle variabelen (OPENAI_API_KEY, GOOGLE_PROJECT_ID, etc.) correct zijn ingesteld in je run_automation.sh script.")
        return None

def load_config():
    """Laadt basisconfiguratie uit config.json."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        print(f"✅ Basisconfiguratie succesvol geladen uit {CONFIG_FILE}")
        return config
    print(f"❌ Fout: {CONFIG_FILE} niet gevonden.")
    return None

def load_templates_from_notion(api_key: str, database_id: str):
    """
    Fetches templates and prompts from a Notion database.
    """
    print("   -> Templates en prompts ophalen uit Notion...")
    try:
        notion = notion_client.Client(auth=api_key)
        response = notion.databases.query(database_id=database_id)
        results = response.get("results")

        templates = {}
        for page in results:
            key_property = page.get("properties", {}).get("Key", {}).get("title", [])
            if not key_property:
                continue
            key = key_property[0].get("plain_text")

            value_property = page.get("properties", {}).get("Value", {}).get("rich_text", [])
            value = value_property[0].get("plain_text", "") if value_property else ""
            
            if key:
                templates[key] = value
        
        print("✅ Templates succesvol geladen uit Notion.")
        return templates
    except Exception as e:
        print(f"❌ Fout bij het laden van templates uit Notion: {e}")
        return None

# --- Helper Functions ---

def find_and_set_header(df, sheet_name):
    """Vindt de header-rij, stelt deze in en geeft de header-rij index terug."""
    print(f"   -> Zoeken naar header-rij in tabblad '{sheet_name}'...")
    for i, row in df.iterrows():
        row_values = list(map(str, row.values))
        if "First Name" in row_values:
            header_index = i
            new_header = df.iloc[header_index].astype(str).str.strip()
            df = df[header_index+1:].copy()
            df.columns = new_header
            
            print("\n--- GEVONDEN KOLOMNAMEN (opgeschoond) ---")
            print(list(df.columns))
            print("----------------------------------------\n")
            
            df['sheet_row_number'] = df.index + 1
            print(f"   ✅ Header gevonden op rij {header_index + 1} in '{sheet_name}'.")
            return df.reset_index(drop=True), header_index + 1
    print(f"   ❌ Fout: Header met 'First Name' niet gevonden in '{sheet_name}'.")
    return None, None

def load_data_from_google_sheet(gc, sheet_url):
    """Leest en verwerkt data uit de Google Sheet."""
    print("\n--- Stap 2: Data ophalen uit Google Sheets ---")
    spreadsheet = gc.open_by_url(sheet_url)
    
    print("   -> Lezen van 'Lead Tracking' tabblad...")
    worksheet = spreadsheet.worksheet('Lead Tracking')
    df_tracking_raw = get_as_dataframe(worksheet, evaluate_formulas=True, header=None)
    
    print("   -> Lezen van 'Invoerblad' tabblad...")
    invoer_worksheet = spreadsheet.worksheet('Invoerblad')
    df_invoer_raw = get_as_dataframe(invoer_worksheet, evaluate_formulas=True, header=None)

    df_tracking, header_row_num = find_and_set_header(df_tracking_raw, 'Lead Tracking')
    df_invoer, _ = find_and_set_header(df_invoer_raw, 'Invoerblad')

    if df_tracking is None or df_invoer is None:
        raise ValueError("Kon de header-rij niet vinden in een van de tabbladen.")

    print("   -> Data van beide tabbladen samenvoegen...")
    df_tracking.columns = df_tracking.columns.str.strip()
    df_invoer.columns = df_invoer.columns.str.strip()
    
    df_tracking['First Name'] = df_tracking['First Name'].str.strip()
    df_tracking['Last Name'] = df_tracking['Last Name'].str.strip()
    df_invoer['First Name'] = df_invoer['First Name'].str.strip()
    df_invoer['Last Name'] = df_invoer['Last Name'].str.strip()
    
    cols_from_invoer = [col for col in ["Keywords", "LinkedIn extracted werkzaamheden"] if col in df_invoer.columns]
    df_tracking = df_tracking.drop(columns=cols_from_invoer, errors='ignore')
    merged_df = pd.merge(df_tracking, df_invoer[["First Name", "Last Name"] + cols_from_invoer], on=["First Name", "Last Name"], how="left")
    
    print("   -> Datums omzetten...")
    merged_df['Last contacted'] = pd.to_datetime(merged_df['Last contacted'], errors='coerce', dayfirst=True)
    print("✅ Data succesvol geladen en voorbereid.")
    return merged_df, header_row_num

def find_leads_to_contact(df):
    """Filtert de DataFrame voor leads die een e-mail nodig hebben."""
    print("\n--- Stap 3: Leads identificeren die een actie vereisen ---")
    today = pd.to_datetime(datetime.now().date())
    
    df_copy = df.copy()

    for i in range(1, 5):
        if f'email {i}' not in df_copy.columns:
            df_copy[f'email {i}'] = ''

    cold_leads = df_copy[(df_copy['Stage'] == 'Cold') & (df_copy['email 1'].fillna('') == '')].copy()
    cold_leads['next_email_col'] = 'email 1'
    cold_leads['next_stage'] = 'Email 1'
    cold_leads['template_to_use'] = 'cold_email_template'
    print(f"   -> {len(cold_leads)} 'Cold' leads gevonden die 'email 1' nodig hebben.")
    
    approaching_df = df_copy[df_copy['Stage'] == 'Approaching'].copy()
    follow_up_leads = pd.DataFrame() 
    
    if not approaching_df.empty:
        approaching_df['last_contacted_date'] = pd.to_datetime(approaching_df['Last contacted'], errors='coerce').dt.normalize()
        approaching_df.dropna(subset=['last_contacted_date'], inplace=True)
        
        if not approaching_df.empty:
            start_dates = approaching_df['last_contacted_date'].values.astype('datetime64[D]')
            end_dates = np.repeat(today.to_numpy().astype('datetime64[D]'), len(approaching_df))
            
            approaching_df['workdays_since_contact'] = np.busday_count(start_dates, end_dates)
            
            follow_up_leads = approaching_df[approaching_df['workdays_since_contact'] >= 4].copy()
            print(f"   -> {len(follow_up_leads)} 'Approaching' leads gevonden die >= 4 werkdagen geleden zijn benaderd.")
            
            def get_next_step(row):
                if row['Stage Approaching'] == 'Email 1' and (pd.isna(row['email 2']) or row['email 2'] == ''):
                    return pd.Series(['email 2', 'Email 2', 'follow_up_2_template'])
                if row['Stage Approaching'] == 'Email 2' and (pd.isna(row['email 3']) or row['email 3'] == ''):
                    return pd.Series(['email 3', 'Email 3', 'follow_up_3_template'])
                if row['Stage Approaching'] == 'Email 3' and (pd.isna(row['email 4']) or row['email 4'] == ''):
                    return pd.Series(['email 4', 'Email 4', 'follow_up_4_template'])
                return pd.Series([None, None, None])

            if not follow_up_leads.empty:
                next_step_data = follow_up_leads.apply(get_next_step, axis=1)
                next_step_data.columns = ['next_email_col', 'next_stage', 'template_to_use']
                follow_up_leads = pd.concat([follow_up_leads, next_step_data], axis=1)
                follow_up_leads.dropna(subset=['next_email_col'], inplace=True)
                print(f"   -> Na controle op reeds verstuurde mails, zijn er {len(follow_up_leads)} follow-ups nodig.")

    all_leads_to_contact = pd.concat([cold_leads, follow_up_leads])
    print(f"✅ Totaal {len(all_leads_to_contact)} leads om te verwerken.")
    return all_leads_to_contact

def generate_email_content(api_key, prompt):
    """Genereert e-mail onderwerp en body via OpenAI."""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=800
    )
    content = response.choices[0].message.content
    email_data = json.loads(content)
    return email_data.get("subject"), email_data.get("body")

def search_last_email_thread(imap_config, recipient_email):
    """Zoekt naar de laatste e-mailthread met een specifieke ontvanger."""
    try:
        mail = imaplib.IMAP4_SSL(imap_config['server'], imap_config['port'])
        mail.login(imap_config['username'], imap_config['password'])
        mail.select('"Sent"') # Use quotes for folder names that might have spaces
        _, data = mail.search(None, f'(TO "{recipient_email}")')
        if not data[0]:
            mail.logout()
            return None
        latest_email_id = data[0].split()[-1]
        _, msg_data = mail.fetch(latest_email_id, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        message_id = msg.get('Message-ID')
        mail.logout()
        return message_id
    except Exception as e:
        print(f"     -> Fout bij zoeken naar thread: {e}")
        return None

def save_follow_up_draft(imap_config, to_email, subject, body, in_reply_to=None, references=None):
    """Slaat een e-mail op als concept, eventueel als antwoord in een thread."""
    try:
        msg = MIMEMultipart()
        msg['To'] = str(to_email)
        msg['Subject'] = str(subject)
        
        if in_reply_to:
            msg['In-Reply-To'] = in_reply_to
            msg['References'] = references
        
        msg.attach(MIMEText(str(body), 'plain', 'utf-8'))
        
        mail = imaplib.IMAP4_SSL(imap_config['server'], imap_config['port'])
        mail.login(imap_config['username'], imap_config['password'])
        mail.append(f'"{imap_config["drafts_folder"]}"', '\\Draft', imaplib.Time2Internaldate(datetime.now(timezone.utc)), msg.as_bytes())
        mail.logout()
        return True, "Concept opgeslagen!"
    except Exception as e:
        return False, f"Fout bij opslaan: {e}"

def update_cells_in_sheet(worksheet, updates):
    """Werkt specifieke cellen in de Google Sheet bij via een batch-update."""
    if not updates:
        print("\n--- Geen updates om door te voeren naar Google Sheet. ---")
        return
        
    print(f"\n--- {len(updates)} cellen worden klaargemaakt voor de update... ---")
    try:
        worksheet.update_cells(updates, value_input_option='USER_ENTERED')
        print(f"✅ Google Sheet succesvol bijgewerkt!")
    except Exception as e:
        print(f"❌ Fout bij het bijwerken van de Google Sheet: {e}")

# --- Main Script ---

def main():
    """Main function to run the email automation script."""
    print(f"--- Start Automatisch Follow-up Script ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    
    print("\n--- Stap 0: Geheimen laden ---")
    secrets = load_secrets_from_env()
    if not secrets:
        print("Script gestopt.")
        return

    print("\n--- Stap 1: Configuratie laden ---")
    config = load_config()
    if not config:
         print("Script gestopt. Configuratie (config.json) niet gevonden.")
         return

    # Load templates from Notion and merge them with the base configuration
    notion_templates = load_templates_from_notion(
        api_key=secrets['notion']['api_key_notion'],
        database_id=secrets['notion']['database_id']
    )
    if not notion_templates:
        print("Script gestopt. Kon templates uit Notion niet laden.")
        return
    
    config.update(notion_templates) # Merge configs

    # Get required values from secrets and the merged config
    openai_api_key = secrets['openai']['api_key']
    email_pass = secrets['email']['password']
    google_creds = secrets['google_credentials']
    email_user = config['email_user']

    imap_config = {
        "server": config['imap_server'], 
        "port": int(config['imap_port']), 
        "username": email_user,
        "password": email_pass, 
        "drafts_folder": config['drafts_folder']
    }

    try:
        print("\nVerbinding maken met Google API...")
        gc = gspread.service_account_from_dict(google_creds)
        
        df, header_row_num = load_data_from_google_sheet(gc, config['gsheet_url'])

        worksheet = gc.open_by_url(config['gsheet_url']).worksheet('Lead Tracking')
        header_row_values = worksheet.row_values(header_row_num)
        
        required_cols = ["email 1", "email 2", "email 3", "email 4", "Onderwerp"]
        for col_name in required_cols:
            if col_name not in df.columns:
                df[col_name] = ''
        
        col_indices = {name: i + 1 for i, name in enumerate(header_row_values)}
        
        leads = find_leads_to_contact(df)

        if leads.empty:
            print("\nGeen leads om te contacteren. Script voltooid.")
            return
        
        print("\n--- Stap 4: E-mails genereren en opslaan ---")
        updates_to_batch = []
        for index, row in leads.iterrows():
            sheet_row_index = int(row['sheet_row_number'])
            
            if pd.isna(row.get('First Name')) or row.get('First Name') == '':
                continue

            print(f"\n▶️ Verwerken van: {row['First Name']} {row['Last Name']} (Sheet rij {sheet_row_index})")
            
            template_name = row['template_to_use']
            next_stage = row['next_stage']
            email_col_to_update = row['next_email_col']

            print(f"   -> Status: '{row['Stage']}', Actie: Genereer '{email_col_to_update}'...")
            
            format_dict = {k: '' if pd.isna(v) else v for k, v in row.to_dict().items()}
            
            try:
                email_body_template = config[template_name]
                formatted_email_body = email_body_template.format(**format_dict)
            except KeyError as e:
                print(f"   ⚠️ WAARSCHUWING: De placeholder '{{{e}}}' in je e-mailtemplate '{template_name}' (Notion) bestaat niet als kolom in je Google Sheet.")
                formatted_email_body = email_body_template

            format_dict['example_email'] = formatted_email_body
            
            if email_col_to_update == 'email 1':
                prompt_template_key = 'Prompt_template_1'
            else:
                prompt_template_key = 'Prompt_template_2'

            try:
                main_prompt_template = config[prompt_template_key]
                prompt = main_prompt_template.format(**format_dict)
            except KeyError as e:
                if str(e.args[0]) == prompt_template_key:
                     print(f"   ❌ FOUT: De template '{prompt_template_key}' is niet gevonden in je Notion database.")
                else:
                     print(f"   ❌ FOUT: De placeholder '{{{e}}}' in je '{prompt_template_key}' (Notion) bestaat niet als kolom in je Google Sheet.")
                print(f"   -> Deze lead wordt overgeslagen.")
                continue
            
            print(f"   -> Volledige prompt voor OpenAI:\n---\n{prompt}\n---")
            
            print("   -> Versturen van prompt naar OpenAI...")
            generated_subject, generated_body = generate_email_content(openai_api_key, prompt)
            print(f"   -> OpenAI antwoord ontvangen. Onderwerp: '{generated_subject}'")

            if email_col_to_update == 'email 1':
                final_subject = generated_subject
                message_id = None
                print("   -> Dit is de eerste e-mail, er wordt geen thread gezocht.")
            else:
                original_subject = row['Onderwerp']
                final_subject = f"Re: {original_subject}" if original_subject else generated_subject
                print(f"   -> Zoeken naar vorige e-mailthread...")
                message_id = search_last_email_thread(imap_config, row['E-mail'])
                if message_id:
                    print("   -> Vorige thread gevonden, e-mail wordt als antwoord toegevoegd.")
                else:
                    print("   -> Geen vorige thread gevonden, e-mail wordt als nieuwe conversatie opgeslagen.")

            print("   -> Concept opslaan in e-mailaccount...")
            success, message = save_follow_up_draft(imap_config, row['E-mail'], final_subject, generated_body, in_reply_to=message_id, references=message_id)
            print(f"   -> Resultaat: {message}")

            if success:
                print("   -> Voorbereiden van updates voor Google Sheet...")
                updates_to_batch.append(gspread.Cell(sheet_row_index, col_indices['Last contacted'], datetime.now().strftime('%d-%m-%Y')))
                updates_to_batch.append(gspread.Cell(sheet_row_index, col_indices['Stage Approaching'], next_stage))
                updates_to_batch.append(gspread.Cell(sheet_row_index, col_indices[email_col_to_update], generated_body))
                
                if email_col_to_update == 'email 1':
                    updates_to_batch.append(gspread.Cell(sheet_row_index, col_indices['Onderwerp'], final_subject))
                    updates_to_batch.append(gspread.Cell(sheet_row_index, col_indices['Stage'], 'Approaching'))

        update_cells_in_sheet(worksheet, updates_to_batch)
        print("\n\n--- Script succesvol voltooid! ---")

    except gspread.exceptions.SpreadsheetNotFound:
        print("\n❌ FATALE FOUT: De Google Sheet is niet gevonden. Controleer de 'gsheet_url' in je config.json.")
    except Exception as e:
        import traceback
        print(f"\n❌ FATALE FOUT: Er is een onverwachte fout opgetreden. Script gestopt.")
        print(f"Foutdetails: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
