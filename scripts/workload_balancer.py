def balance_workload(tasks_df, members):
    tasks_sorted = tasks_df.sort_values(by="Predicted Priority", ascending=False)
    
    
    assignments = {member: [] for member in members}
    i = 0
    for idx, task in tasks_sorted.iterrows():
        member = members[i % len(members)]
        assignments[member].append(task['Task'])
        i += 1
    
    return assignments


import pandas as pd


tasks_df = pd.DataFrame({
    "Task": ["Build model", "Clean data", "Visualize results", "Deploy", "Report"],
    "Predicted Priority": [2, 1, 3, 2, 1] # 3 High, 1 Low
})

members = ["Anuska", "Rajiv", "Diya"]
assigned = balance_workload(tasks_df, members)

for person, tasks in assigned.items():
    print(f"{person}: {tasks}")
