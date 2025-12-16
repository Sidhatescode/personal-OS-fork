# personal-os
A simple Python script to categorize your notes,thoguhts,ideas,reflections etc with AI + a custom prompt you set.

ignore below (to be changed)

1. `git clone https://github.com/farzaa/stupid-emails.git`
2. `cd stupid-emails`
2. Rename `secrets.template.json` to `secret.json` and add your OpenAI key in to it.
3. You need a GMail API key. It's not that hard. [Here's](https://chatgpt.com/share/67e5a57f-b544-8013-891b-eebe58a0b6b6) how.
4. Go to `prompt.txt` and add your name + bio.
5. `pip install -r requirements.txt`
6. Run script `python tag.py`, Google Auth will open and ask for read/write acceess. The script takes your email, send it to OpenAI, and gets back a tag. That's it. Nothing is saved. It's not gonna randomly send emails on your behalf lol. You can read through the code if you don't trust me.
