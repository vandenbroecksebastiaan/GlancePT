import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailClient:
    def __init__(self):
        self.smtp_server = "smtp.zoho.eu"
        self.smtp_port = 465
        self.sender_email = "glancept@zohomail.eu"
        self.sender_password = "emailclientpassword" # Yes, I know this is a bad practice
        
        self.base_html = open("mail.html").readlines()
        
    def make_email(self, paper_titles, paper_summaries):
        html = self.base_html
        for idx, line in enumerate(html):
            if "<!-- PAPER SUMMARY -->" in line:
                summary_idx = idx
            
        for idx, (title, summary) in enumerate(zip(paper_titles, paper_summaries)):
            p_tag_title = f'<p style="font-family: sans-serif; font-size: 20px; font-weight: 600; margin: 0; margin-bottom: 5px;">{title}</p>'
            p_tag_summary = f'<p style="font-family: sans-serif; font-size: 14px; font-weight: normal; margin: 0; margin-bottom: 15px;">{summary}</p>'
            html.insert(summary_idx+idx, p_tag_title+p_tag_summary)
        
        self.html = "".join(html)

    def send_email(self, recipient_email):
        # Create a multipart message object
        msg = MIMEText(self.html, 'html')
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Papers With Code update"

        server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
        server.login(self.sender_email, self.sender_password)
        server.send_message(msg)
        server.quit()

