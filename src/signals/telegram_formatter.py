
def format_signal(signal):

    emoji = "🟢" if signal["signal"] == "BUY" else "🔴"

    return f"""
🔥 XAUUSD AI SIGNAL

Direction:
{emoji} {signal['signal']}

Confidence:
{signal['confidence'] * 100:.1f}%

Entry:
{signal['entry_price']}

Current Price:
{signal['current_price']}

Stop Loss:
{signal['stop_loss']}

Take Profit:
{signal['take_profit']}

Risk Reward:
1:{signal['risk_reward_ratio']}

AI Model:
XAUUSD Scalping Bot
"""
