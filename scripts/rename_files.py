import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            parts = filename.split('_')

            # Kontrollera om filnamnet följer mönstret och behöver ändras
            if len(parts) > 2 and not parts[1].isdigit():
                new_name = parts[0] + '_' + parts[1] + '_' + 'monthly-' + '-'.join(parts[2:])
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_name)

                # Byt namn på filen
                os.rename(old_file, new_file)
                print(f"Renamed '{filename}' to '{new_name}'")

# Ange sökvägen till mappen med filerna
directory_path = 'data/features/rename_data'
rename_files(directory_path)
