import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

FILE_NAME = "students.csv"
SUBJECTS = ["English", "Maths", "Hindi", "Science", "Social Science"]

# ------------------ Helper Functions ------------------

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def file_exists():
    return os.path.exists(FILE_NAME)

def read_students():
    """Reads the CSV into a pandas DataFrame, ensuring term column exists."""
    if not file_exists():
        columns = ["term", "roll", "first_name", "middle_name", "last_name"] + SUBJECTS
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(FILE_NAME)
    required = ["term", "roll", "first_name", "middle_name", "last_name"] + SUBJECTS
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df = df[required]
    return df

def write_students(df):
    df.to_csv(FILE_NAME, index=False)

def get_full_name(row):
    parts = [row.get("first_name", ""), row.get("middle_name", ""), row.get("last_name", "")]
    return " ".join([p for p in parts if isinstance(p, str) and p.strip()])

def get_latest_term(df):
    """Returns latest numeric or string term value."""
    if "term" not in df.columns or df["term"].dropna().empty:
        return None
    term_nums = pd.to_numeric(df["term"], errors="coerce")
    if term_nums.notna().any():
        return str(int(term_nums.max()))
    return str(df["term"].astype(str).iloc[-1])

def banner(title):
    print(Fore.CYAN + Style.BRIGHT + f"\n==== {title} ====\n" + Style.RESET_ALL)

# ------------------ Core Functionalities ------------------

def add_student():
    clear_screen()
    banner("Add Student Record")
    df = read_students()

    term = input("Enter term (e.g., 1, 2, 3): ").strip()
    first = input("Enter first name: ").strip()
    middle = input("Enter middle name (press enter if none): ").strip()
    last = input("Enter last name: ").strip()

    marks = {sub: input(f"Enter marks for {sub}: ").strip() for sub in SUBJECTS}

    full_name = f"{first} {middle} {last}".replace("  ", " ").strip().lower()
    existing = df[df.apply(get_full_name, axis=1).str.lower() == full_name]
    roll = existing.iloc[0]["roll"] if not existing.empty else str(len(df["roll"].dropna()) + 1)

    new_student = {"term": term, "roll": roll, "first_name": first, "middle_name": middle, "last_name": last, **marks}
    df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
    write_students(df)

    print(Fore.GREEN + "✅ Student added successfully!\n")

def display_students():
    clear_screen()
    banner("All Students")

    df = read_students()
    if df.empty:
        print(Fore.RED + "No data present.\n")
        return

    df["Full Name"] = df.apply(get_full_name, axis=1)
    latest = get_latest_term(df)
    snapshot = df[df["term"].astype(str) == latest] if latest else df.copy()
    snapshot = snapshot.drop_duplicates(subset=["Full Name"])
    snapshot = snapshot.sort_values("Full Name", key=lambda x: x.str.lower())

    table = [(r["roll"], r["Full Name"]) for _, r in snapshot.iterrows()]
    print(tabulate(table, headers=["Roll No", "Full Name"], tablefmt="pretty"))
    print(Fore.YELLOW + f"\n(Showing records from Term {latest})\n" if latest else "")

def search_student():
    clear_screen()
    banner("Search Student")
    df = read_students()
    if df.empty:
        print(Fore.RED + "No data present.\n")
        return
    df["Full Name"] = df.apply(get_full_name, axis=1)
    key = input("Search by roll or full name: ").strip().lower()
    matches = df[(df["roll"].astype(str) == key) | (df["Full Name"].str.lower() == key)]
    if matches.empty:
        print(Fore.RED + "❌ Student not found.\n")
        return
    term_nums = pd.to_numeric(matches["term"], errors="coerce")
    chosen = matches.loc[term_nums.idxmax()] if term_nums.notna().any() else matches.iloc[-1]

    print(Fore.CYAN + f"\n--- Student Details (Term {chosen['term']}) ---")
    print(f"Roll : {chosen['roll']}")
    print(f"Name : {chosen['Full Name']}\n")

    marks = [(s, chosen[s]) for s in SUBJECTS]
    print(tabulate(marks, headers=["Subject", "Marks"], tablefmt="pretty"))

    scores = np.array([int(chosen[s]) for s in SUBJECTS])
    avg = np.mean(scores)
    strong, weak = SUBJECTS[np.argmax(scores)], SUBJECTS[np.argmin(scores)]
    print()
    print(tabulate([
        [Fore.GREEN + "Average Marks", f"{avg:.2f}"],
        [Fore.BLUE + "Strong Subject", strong],
        [Fore.RED + "Weak Subject", weak]
    ], tablefmt="pretty"))
    print()

def class_analytics():
    clear_screen()
    banner("Class Analytics")

    df = read_students()
    if df.empty:
        print(Fore.RED + "No data present.\n")
        return
    df[SUBJECTS] = df[SUBJECTS].apply(pd.to_numeric, errors="coerce")
    df["Full Name"] = df.apply(get_full_name, axis=1)
    terms = sorted(df["term"].dropna().astype(str).unique())
    choice = input(f"Analyze which term? Options: {terms}, 'all' for all terms, Enter for latest: ").strip()
    term_to_use = get_latest_term(df) if choice == "" else choice

    if choice.lower() == "all":
        per_student = df.groupby("Full Name")[SUBJECTS].mean()
        scope = "Across All Terms"
    else:
        per_student = df[df["term"].astype(str) == str(term_to_use)].groupby("Full Name")[SUBJECTS].mean()
        scope = f"Term {term_to_use}"

    subject_avg = per_student.mean()
    subject_std = per_student.std()
    per_student["Average"] = per_student.mean(axis=1)
    per_student["Rank"] = per_student["Average"].rank(ascending=False, method="dense").astype(int)

    best, worst = per_student["Average"].idxmax(), per_student["Average"].idxmin()

    print(Fore.YELLOW + f"\n--- Class Analytics ({scope}) ---")
    print(Fore.CYAN + f"Number of Students: {len(per_student)}\n")

    print(tabulate([(s, f"{subject_avg[s]:.2f}") for s in SUBJECTS],
                   headers=["Subject", "Avg Marks"], tablefmt="pretty"))
    overall = [
        ["Overall Avg", f"{subject_avg.mean():.2f}"],
        ["Best Performer", f"{best} ({per_student.loc[best,'Average']:.2f})"],
        ["Worst Performer", f"{worst} ({per_student.loc[worst,'Average']:.2f})"],
        ["Strongest Subject", subject_avg.idxmax()],
        ["Weakest Subject", subject_avg.idxmin()],
    ]
    print()
    print(tabulate(overall, tablefmt="pretty"))
    print(Fore.MAGENTA + "\n--- Student Rankings ---")
    rank_tbl = per_student.reset_index().sort_values("Rank")[["Rank", "Full Name", "Average"]]
    print(tabulate(rank_tbl.values.tolist(), headers=["Rank", "Full Name", "Average"], tablefmt="pretty"))
    print()

def update_student():
    clear_screen()
    banner("Update Student Record")
    df = read_students()
    if df.empty:
        print(Fore.RED + "No data present.\n")
        return
    df["Full Name"] = df.apply(get_full_name, axis=1)
    key = input("Search by roll or full name: ").strip().lower()
    matches = df[(df["roll"].astype(str) == key) | (df["Full Name"].str.lower() == key)]
    if matches.empty:
        print(Fore.RED + "❌ Student not found.\n")
        return
    term_nums = pd.to_numeric(matches["term"], errors="coerce")
    idx = term_nums.idxmax() if term_nums.notna().any() else matches.index[-1]
    choice = input("What do you want to update? (name/marks): ").strip().lower()
    if choice == "name":
        part = input("Which part? (first/middle/last): ").strip().lower()
        if part in ["first", "middle", "last"]:
            new_val = input(f"Enter new {part} name: ").strip()
            df.loc[idx, f"{part}_name"] = new_val
            print(Fore.GREEN + "✅ Name updated.\n")
    elif choice == "marks":
        print("Subjects:", ", ".join(SUBJECTS))
        sub = input("Enter subject to update: ").strip()
        if sub not in SUBJECTS:
            print(Fore.RED + "❌ Invalid subject.\n")
            return
        df.loc[idx, sub] = input(f"Enter new marks for {sub}: ").strip()
        print(Fore.GREEN + "✅ Marks updated.\n")
    else:
        print(Fore.RED + "❌ Invalid choice.\n")
        return
    write_students(df)

def delete_student():
    clear_screen()
    banner("Delete Student Record")
    df = read_students()
    if df.empty:
        print(Fore.RED + "No data present.\n")
        return
    df["Full Name"] = df.apply(get_full_name, axis=1)
    key = input("Search by roll or full name: ").strip().lower()
    matches = df[(df["roll"].astype(str) == key) | (df["Full Name"].str.lower() == key)]
    if matches.empty:
        print(Fore.RED + "❌ Student not found.\n")
        return
    print("\nFound records:")
    print(tabulate(matches[["term", "roll", "Full Name"]].values.tolist(),
                   headers=["Term", "Roll", "Full Name"], tablefmt="pretty"))
    choice = input("Delete (l)atest term only, (a)ll terms, (c)ancel [l/a/c]: ").strip().lower()
    if choice == "c":
        return
    if choice == "a":
        df = df.drop(matches.index)
        print(Fore.GREEN + "✅ All records deleted.\n")
    else:
        term_nums = pd.to_numeric(matches["term"], errors="coerce")
        idx_drop = term_nums.idxmax() if term_nums.notna().any() else matches.index[-1]
        df = df.drop(index=idx_drop)
        print(Fore.GREEN + "✅ Latest term record deleted.\n")
    write_students(df)

def trend_tracking():
    clear_screen()
    banner("Trend Tracking")
    df = read_students()
    if df.empty or "term" not in df.columns:
        print(Fore.RED + "No term-based data found.\n")
        return
    df[SUBJECTS] = df[SUBJECTS].apply(pd.to_numeric, errors="coerce")
    df["Average"] = df[SUBJECTS].mean(axis=1)
    df["Full Name"] = df.apply(get_full_name, axis=1)
    pivot = df.pivot_table(index="Full Name", columns="term", values="Average", aggfunc="mean").fillna(0).reset_index()
    cols = ["Full Name"] + sorted([c for c in pivot.columns if c != "Full Name"])
    pivot = pivot[cols]
    
    if len(cols) > 2:
        pivot["Change"] = pivot[cols[-1]] - pivot[cols[1]]
        pivot["Trend"] = np.where(
            pivot["Change"] > 0, "Improved",
            np.where(pivot["Change"] < 0, "Declined", "No Change")
        )

    # --- Apply color formatting for Trend ---
    def color_trend(val):
        if val == "Improved":
            return Fore.GREEN + val + Style.RESET_ALL
        elif val == "Declined":
            return Fore.RED + val + Style.RESET_ALL
        elif val == "No Change":
            return Fore.YELLOW + val + Style.RESET_ALL
        return val

    if "Trend" in pivot.columns:
        pivot["Trend"] = pivot["Trend"].apply(color_trend)

    print(Fore.YELLOW + "\n--- Student Performance Trend ---" + Style.RESET_ALL)
    print(tabulate(pivot.round(2).values.tolist(), headers=pivot.columns, tablefmt="pretty"))

    print(Fore.YELLOW + "\n--- Class Average Trend ---" + Style.RESET_ALL)
    class_trend = df.groupby("term")["Average"].mean().reset_index().sort_values("term")
    print(tabulate(class_trend.round(2).values.tolist(), headers=["Term", "Class Avg"], tablefmt="pretty"))
    print()

# ------------------ Main Program ------------------

def main():
    while True:
        clear_screen()
        banner("Student Analytics System")
        print("1. Add Student")
        print("2. Display All Students")
        print("3. Search Student")
        print("4. Show Class Analytics")
        print("5. Update Student Record")
        print("6. Delete Student")
        print("7. Show Trend Tracking")
        print("8. Exit")

        choice = input(Fore.YELLOW + "\nChoose an option: ").strip()
        if choice == "1": add_student()
        elif choice == "2": display_students()
        elif choice == "3": search_student()
        elif choice == "4": class_analytics()
        elif choice == "5": update_student()
        elif choice == "6": delete_student()
        elif choice == "7": trend_tracking()
        elif choice == "8":
            clear_screen()
            print(Fore.CYAN + "Exiting... Goodbye!")
            break
        else:
            print(Fore.RED + "❌ Invalid option.\n")
        input(Fore.YELLOW + "\nPress Enter to return to main menu...")

if __name__ == "__main__":
    main()
