# dissertation
Configurations can be found on the `config.py` file.
Use the following command to install the required packages:
```bash
conda create --name <env> --file requirements.txt
```

To Generate the simulated data, run the following command:
```bash
python3 generate_data.py
```
To train the model, run the following command:
```bash
python3 main.py
```
To run the inference, run the following command:
```bash
python3 inference.py
```
