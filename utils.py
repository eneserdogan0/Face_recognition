from tabulate import tabulate

def get_person_name(person_path):
    return person_path.split('_')[-1].split('\\')[-2]

def print_report(title, headers, data):
    print(f"\n{title}\n")
    print(tabulate(data, headers=headers, tablefmt="pretty"))

def save_report_to_file(filename, title, headers, data):
    with open(filename, "a") as f:
        f.write(f"\n{title}\n")
        f.write(tabulate(data, headers=headers, tablefmt="pretty"))
        f.write("\n")
