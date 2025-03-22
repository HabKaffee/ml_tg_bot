# Telegram bot

## Prepare environment
1. Set up virtualenv
```bash
python3.10 -m venv venv
source venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Download sticker preparation model checkpoint
```bash
mkdir -p models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Start the bot
First of all, you need to get bot token from `@BotFather` telegram-bot.  

Once the token is aquired, set BOT_TOKEN enviroment variable with provided token.  

Assuming you set the token in .env file, run the following commands
```bash
source .env
python app.py
```

First launch will take some time because other models will be downloaded if missing

