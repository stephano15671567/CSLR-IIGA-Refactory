import numpy as np
import matplotlib.pyplot as plt
import os

# RUTA ACTUAL DE TU LOG (sacada de tus logs)
ruta_npy = r'D:\IIGA-Tesis\CSLR-IIGA\trained_model_FINAL\Abril_Entrenamiento_Real\2026-04-18-04.40\learning_curves.npy'

if not os.path.exists(ruta_npy):
    print(f"Error: No se encuentra el archivo en {ruta_npy}")
else:
    # Cargar datos. En tu train.py se guarda como un diccionario directamente.
    data = np.load(ruta_npy, allow_pickle=True).item()
    
    # Extraer usando las llaves EXACTAS de tu train.py
    wer = [w * 100 for w in data['wer']]  # Convertimos 0.56 a 56.0
    train_loss = data['train_losses']     # Nota la 's'
    val_loss = data['val_losses']         # Nota la 's'
    epochs = range(len(wer))

    # --- GRÁFICO 1: WER ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, wer, color='#1f77b4', marker='o', linewidth=2, label='WER Sistema Propuesto')
    plt.title('Evolución del Word Error Rate (WER)')
    plt.xlabel('Época')
    plt.ylabel('WER %')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafico_wer_actual.png', dpi=300)
    print("✓ Generado: grafico_wer_actual.png")

    # --- GRÁFICO 2: LOSS ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, color='blue', label='Train Loss')
    plt.plot(epochs, val_loss, color='orange', label='Val Loss')
    plt.title('Curva de Pérdida (Loss) - IIGA Optimizado')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (CTC)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafico_loss_actual.png', dpi=300)
    print("✓ Generado: grafico_loss_actual.png")

    print(f"\nResumen última época ({len(epochs)-1}):")
    print(f"- WER: {wer[-1]:.2f}%")
    print(f"- Train Loss: {train_loss[-1]:.4f}")