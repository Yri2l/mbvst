from tqdm import tqdm
import os
import re
import shutil
import re
from datetime import datetime


from PLNE_exp import PLNE_exp
from PLNE_CP import PLNE_CP
from PLNE_CPM import PLNE_CPM
from PLNE_Martin import PLNE_Martin
from PLNE_Cycle import PLNE_cycle_final
from Outils import save_to_csv, graphiques

def parse_instance_name(name):
    """Extracts the numerical parts from the instance file name."""
    match = re.match(r"Spd_RF2_(\d+)_(\d+)_(\d+)", name)
    if match:
        return tuple(map(int, match.groups()))
    return None

def main(methode, dossier):
    liste_fichier = []
    for fichier in os.listdir(f"./Instances/{dossier}/"):
        if fichier[:3] == "Spd":
            liste_fichier.append(fichier)

    # Trier les instances de la plus petite à la plus grande
    liste_fichier.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
    # print(liste_fichier)

    # ind = 0
    # for k in range(len(liste_fichier)):
    #     if liste_fichier[k][:-4] == "Spd_RF2_350_492_4579":
    #         ind = k
    #         break

    # print(f"Nombre d'instances à traiter: {len(liste_fichier) - ind}.")
    # indice 39 pour le CPM
    print(f"Nombre d'instances à traiter: {len(liste_fichier)}.")
    # for i in range(ind, len(liste_fichier)):
    ind = 0
    print(f"Nombre d'instances à traiter: {len(liste_fichier) - ind}.")
    # indice 121 pour le CPM
    # print(f"Nombre d'instances à traiter: {len(liste_fichier)}.")
    for i in range(ind, len(liste_fichier)):
        # for i in range(200):
        path_instance = f"{dossier}/{liste_fichier[i]}"
        date = datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
        print(f"[{date}] {i + 1}/{len(liste_fichier)}: {path_instance}")
        if not os.path.exists(f"./Solutions/{dossier}/{liste_fichier[i][:-4]}/sol_{methode}_{liste_fichier[i]}"):
            if methode == "PLNE_Exp":
                PLNE_exp(instance=path_instance, verbose=False)
            elif methode == "PLNE_CP":
                PLNE_CP(instance=path_instance, verbose=False)
            elif methode == "PLNE_CPM":
                PLNE_CPM(instance=path_instance, verbose=False)
            elif methode == "PLNE_Martin":
                PLNE_Martin(instance=path_instance, verbose=False)
            elif methode == "PLNE_Cycle4_bis_preprocessing":
                try:
                    PLNE_cycle_final(instance=path_instance, maxiter=1e4, preprocessing=True, verbose=False)
                except:
                    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    methode = "PLNE_Cycle4_bis_preprocessing"
    dossier = "Spd_Inst_Rid_Final2"
    # tqdm.disable = True
    main(methode=methode, dossier=dossier)
    # for instance in os.listdir(f"./Solutions/{dossier}/"):
    #     for fichier in os.listdir(f"./Solutions/{dossier}/{instance}/"):
    #         if fichier[9:12] == "CP_":
    #             print(f"./Solutions/{dossier}/{instance}/{fichier}")
    #             print(f"./Solutions/{dossier}/{instance}/{fichier[:9]}Optimal{fichier[11:]}")
    #             os.rename(f"./Solutions/{dossier}/{instance}/{fichier}", f"./Solutions/{dossier}/{instance}/{{fichier}[:9]}Optimal{{fichier}[11:]}")
    # save_to_csv(dossier=dossier)
    # graphiques(dossier=dossier)
    # valeurs_stats(dossier=dossier)

