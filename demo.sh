#!/data/data/com.termux/files/usr/bin/bash
#
# Demo visual de Cerebro Zero — para grabar GIF
# Duración: ~20 segundos
#

cd ~/cerebro_zero

# Pausa dramática inicial
sleep 1

# ============================================
# 1) INFO DEL SISTEMA (5 segundos)
# ============================================
clear
python cli.py info
sleep 3

# ============================================
# 2) RL EN ACCIÓN (8 segundos)
# ============================================
clear
python training/reinforcement.py
sleep 4

# ============================================
# 3) CEREBRO V3 (5 segundos)
# ============================================
clear
python agent/cerebro_v3.py
sleep 3

# ============================================
# FINAL
# ============================================
clear
echo ""
echo "════════════════════════════════════════════════"
echo "  🧠 CEREBRO ZERO 4.0"
echo "  Autonomous Learning Cognitive Agent"
echo "  742 tests · 61 fases · Sin frameworks"
echo "════════════════════════════════════════════════"
echo ""
echo "  github.com/lecodev-26/cerebro-zero"
echo ""
sleep 3
