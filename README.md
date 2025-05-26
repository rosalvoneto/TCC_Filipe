TCC Docking Molecular

## Ajustes do Ambiente 🛠️

1. Instalação do Python 3.9.0 🐍
2. Execute `pip install -r requirements.txt` para instalar dependências📦
3. Digite no terminal src/models para ir para o diretório adequado. 

## Treinando usando LSTM 🧠

1. O formato adequando da do arquivo está em datasets. Como exemplo use: `dados_lstm_5ht1b.csv`.

#### Comando
```bash
python main_lstm.py --input dados_lstm_5ht1b --descriptors morgan_onehot_mac --training_sizes 1400 --cross_validation True 
```

## Usando MLP 🌳
1. O formato adequando da do arquivo está em datasets. Como exemplo use: `dados_mlp_5ht1b.csv`.
2. Prepare o dataset com o comando abaixo
```bash
python create_fingerprint_data.py --input dados_mlp_5ht1b --descriptors morgan_onehot_mac
```
3. Execução do Treinamento
```bash
python main_ml.py --input dados_mlp_5ht1b --descriptors morgan_onehot_mac --training_sizes 1400 --regressor mlp
```
