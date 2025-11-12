# Bitget Futures Trading Bot

Automated 24/7 cryptocurrency futures trading bot.

## Setup
1. Deploy to Railway
2. Set environment variables
3. Monitor via logs

## Environment Variables Required
- BITGET_API_KEY
- BITGET_API_SECRET
- BITGET_PASSPHRASE
- TRADING_SYMBOL (default: BTC/USDT:USDT)
- RISK_PERCENTAGE (default: 10)
```

---

### **Step 3: Initialize Git & Push to GitHub**

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Bitget futures trading bot"

# Create GitHub repo and push (using GitHub CLI)
gh auth login
gh repo create bitget-trading-bot --public --source=. --remote=origin --push

# OR manually:
# 1. Create repo on github.com
# 2. Then run:
git remote add origin https://github.com/YOUR_USERNAME/bitget-trading-bot.git
git branch -M main
git push -u origin main
