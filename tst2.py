import subprocess

def calculate_weights(data_path):
    # Commande à exécuter
    command = f"python3 ./tools/calculate_weights.py --data_path {data_path}"

    # Exécution de la commande
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("Output:", result.stdout)
        print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Command output: {e.output}")

# Spécifiez le chemin vers les données de segmentation
data_path = "/chemin/vers/les/donnees/segmentation"

# Calcul des poids
calculate_weights(data_path)
