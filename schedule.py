from __future__ import division
from __future__ import print_function
from os.path import join
from datetime import datetime
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

wdir = "C:\\Users\\fouta\\Google Drive\\csu\\ssc_data_analysis\\scheduling"


def main():
    # Load preferences and demand (simulate for now)
    preferences_raw = pd.read_excel(join(wdir, "SSC Scheduling - Spring 2019.xlsx"), "Preferences")
    shiftdemand_raw = pd.read_excel(join(wdir, "SSC Scheduling - Spring 2019.xlsx"), "ShiftDemand")
    assignments_raw = pd.read_excel(join(wdir, "SSC Scheduling - Spring 2019.xlsx"), "Assignments")

    # Format data for optimization
    shift_demands = shiftdemand_raw["Demand"].values
    shifts = (shiftdemand_raw["Day"].fillna(method='ffill') + " " + shiftdemand_raw["Time"]).values
    tutors = assignments_raw["Tutor"].values
    assignments = assignments_raw["Assignment"].values * 2  # Half hour segments

    preferences = preferences_raw.iloc[1:-4, 2:-3]
    pref = 1*(preferences == "Preferred").values
    unavail = 1*(preferences == "Unavailable").values
    notpref = 1*(preferences == "Not Preferred").values
    neutral = 1*pd.isna(preferences).values

    larger_blocks = ["Katie Zagnoli", "David Clancy", "Nathan Ryder", "Connor Gibbs"]

    # for testing
    test1 = False
    if test1:
        tutors = tutors[:5]
        shifts = shifts[:]
        assignments = assignments[:len(tutors)]
        #assignments = np.ones(len(assignments[:len(tutors)]), dtype=np.int)*2
        shift_demands = shift_demands[:len(shifts)]
        pref = pref[:len(shifts), :len(tutors)]
        unavail = unavail[:len(shifts), :len(tutors)]
        neutral = neutral[:len(shifts), :len(tutors)]
        notpref = notpref[:len(shifts), :len(tutors)]

    test2 = False
    if test2:
        unavail = np.zeros((len(shifts), len(tutors)), dtype=np.int)

    # Double check things look okay
    print(tutors)
    print(shifts)
    print(assignments)
    print(shift_demands)

    # Create the model.
    model = cp_model.CpModel()

    # Create shift variables.
    schedule = {}
    for tutor in tutors:
        for shift in shifts:
            schedule[(tutor, shift)] = model.NewBoolVar(f'shift_{tutor}_{shift}')

    # Cover demand for each shift (as much as possible)
    const1 = True
    if(const1):
        shift_slack = {}
        # Create slack variables for not satisfying shifts
        for j, shift in enumerate(shifts):
            shift_slack[shift] = model.NewIntVar(0, min(2, shift_demands[j]-2), f'slack_{shift}')
            #shift_slack[shift] = model.NewIntVar(0, 3, f'slack_{shift}')

        # try to have as many tutors on each shift to cover the demand (use slack vars to make up difference)
        for j, shift in enumerate(shifts):
            model.Add(sum(schedule[(tutor, shift)] for tutor in tutors) == shift_demands[j] - shift_slack[shift])

    # Each tutor takes number of shifts appropriate for their assignment (no slack)
    const2b = True
    if(const2b):
        for i, tutor in enumerate(tutors):
            model.Add(sum(schedule[(tutor, shift)] for shift in shifts) == assignments[i])

    # Tutors can't be scheduled when they're unavailable
    const3 = True
    if(const3):
        for i, tutor in enumerate(tutors):
            for j, shift in enumerate(shifts):
                model.Add(schedule[(tutor, shift)] + unavail[j][i] <= 1)

    # limit number of "ends of shift" (encourages contiguity)
    const4 = False
    if(const4):
        # slack variable for reducing number of shift ends (encourages contiguity)
        shift_pos_slack = {}
        shift_neg_slack = {}
        for tutor in tutors:
            for j in range(len(shifts)-1):
                shift_pos_slack[(tutor, shifts[j])] = model.NewIntVar(0, 1, f'slack_shift_pos_{tutor}_{shifts[j]}')
                shift_neg_slack[(tutor, shifts[j])] = model.NewIntVar(0, 1, f'slack_shift_neg_{tutor}_{shifts[j]}')

        # Force capture positive and negative parts of the shifts
        for tutor in tutors:
            for j in range(len(shifts)-1):
                model.Add(schedule[(tutor, shifts[j])] - schedule[(tutor, shifts[j+1])] == shift_pos_slack[(tutor, shifts[j])] - shift_neg_slack[(tutor, shifts[j])])

        # force the positive shifts to be no greater than some number
        for i, tutor in enumerate(tutors):
            model.Add(sum(shift_pos_slack[(tutor, shifts[j])] for j in range(len(shifts)-1)) <= 5)  # fixed number
            #model.Add(sum(shift_pos_slack[(tutor, shifts[j])] for j in range(len(shifts)-1)) <= int(assignments[i]/3))  # fraction of assigned hours


    const5 = True
    if(const5):
        # Duration should be at least one hour
        duration_constraints = {}
        for i, tutor in enumerate(tutors):
            model.Add(schedule[(tutor, shifts[1])] == 1).OnlyEnforceIf(schedule[(tutor, shifts[0])])
            for j in range(2, len(shifts)):
                if unavail[j-1][i] == 0:  # only add constraint if tutor is available for previous shift
                    s = schedule[(tutor, shifts[j])]
                    sm1 = schedule[(tutor, shifts[j-1])]
                    sm2 = schedule[(tutor, shifts[j-2])]
                    b = model.NewBoolVar(f'duration_const_{tutor}_{shifts[i]}')
                    # create b = (sm2 < sm2). If b=True, then this is start of interval so we want constraint to hold
                    model.Add(sm2 < sm1).OnlyEnforceIf(b)
                    model.Add(sm2 >= sm1).OnlyEnforceIf(b.Not())
                    model.Add(s == 1).OnlyEnforceIf(b)
                    duration_constraints[(tutor, shifts[j])] = b

    # limit number of "ends of shift" (encourages contiguity) (alternative method)
    const4b = True
    if(const4b and const5):
        for i, tutor in enumerate(tutors):
            max_blocks = 3 if tutor in larger_blocks else 5
            model.Add(sum(duration_constraints[(tutor, shift)] for shift in shifts
                                                               if (tutor, shift) in duration_constraints.keys()) <= max_blocks)

    # objective function to be maximized
    def obj_function():
        score = 0
        if(const1):
            # maximize fulfillment of shifts
            score = score - sum([shift_slack[shift] for shift in shifts])
        # reward scheduling people's preferences
        score = score + 10*sum(schedule[(tutors[j], shifts[i])] for j in range(len(tutors)) for i in range(len(shifts)) if pref[i][j] == 1)
        # penalize scheduling people's non-preferences
        score = score - 10*sum(schedule[(tutors[j], shifts[i])] for j in range(len(tutors)) for i in range(len(shifts)) if notpref[i][j] == 1)
        return score

    model.Maximize(obj_function())

    # Create the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1200.0  # 20 minutes
    #solver.parameters.max_time_in_seconds = 300  # 5 minutes
    #solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)
    print("Status:", solver.StatusName(status))
    if status == cp_model.INFEASIBLE or status == cp_model.UNKNOWN:
        exit()
    print(f'  - Objective Value = {solver.ObjectiveValue()}')
    print('  - wall time       : %f ms' % solver.WallTime())

    # Write results to excel file
    results_df = pd.DataFrame(columns=tutors, index=shifts)

    print("Writing excel file")

    schedule_matrix = np.zeros((len(shifts), len(tutors)))

    for i, tutor in enumerate(tutors):
        for j, shift in enumerate(shifts):
            val = solver.Value(schedule[(tutor, shift)])
            schedule_matrix[j, i] = val
            pre_entry = preferences.iloc[j, i]
            if pd.isna(pre_entry):
                entry = ""
            elif pre_entry == "Preferred":
                entry = "P"
            elif pre_entry == "Not Preferred":
                entry = "N"
            elif pre_entry == "Unavailable":
                entry = "U"
            else:
                print("Unrecognized preference:", pre_entry)
            if val:
                entry += " (S)"
            results_df.loc[shift, tutor] = entry

    # Summarize columns
    results_df["total"] = np.sum(schedule_matrix, axis=1)
    results_df["demand"] = shift_demands
    results_df["understaffed"] = results_df["demand"] - results_df["total"]

    # interesting data at bottom
    null_buffer = np.array([np.nan, np.nan, np.nan])
    results_df.loc["unavailable"] = np.hstack((np.sum(unavail, axis=0), null_buffer))
    results_df.loc["preferred"] = np.hstack((np.sum(pref, axis=0), null_buffer))
    results_df.loc["preferred_scheduled"] = np.hstack((np.sum(pref*schedule_matrix, axis=0), null_buffer))
    results_df.loc["not_preferred"] = np.hstack((np.sum(notpref, axis=0), null_buffer))
    results_df.loc["not_preferred_scheduled"] = np.hstack((np.sum(notpref*schedule_matrix, axis=0), null_buffer))
    results_df.loc["neutral"] = np.hstack((np.sum(neutral, axis=0), null_buffer))
    results_df.loc["neutral_scheduled"] = np.hstack((np.sum(neutral*schedule_matrix, axis=0), null_buffer))
    results_df.loc["assigned"] = np.hstack((assignments, null_buffer))
    results_df.loc["scheduled"] = np.hstack((np.sum(schedule_matrix, axis=0), null_buffer))
    results_df.loc["overscheduled"] = np.hstack((np.sum(schedule_matrix, axis=0)-assignments, null_buffer))

    # open file for writing
    output_file_path = join(wdir, "schedule_solution_" + datetime.now().strftime("%y%m%d%H%M%S") + ".xlsx")
    writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')
    results_df.to_excel(writer, sheet_name="Schedule")

    # Conditional formatting
    U_format = writer.book.add_format({"font_color": "#000000"})
    P_format = writer.book.add_format({"font_color": "#00ff00"})
    N_format = writer.book.add_format({"font_color": "#ff0000"})
    S_format = writer.book.add_format({"bg_color": "#000000", "bold": 1})
    #S_format = writer.book.add_format("bold": 1, "italic": 1, "border": 2, "border_color": "#000000"})
    writer.sheets["Schedule"].conditional_format("B2:AA105", {"type": "text", "criteria": "begins with",
                                                              "value": "U", "format": U_format})
    writer.sheets["Schedule"].conditional_format("B2:AA105", {"type": "text", "criteria": "begins with",
                                                              "value": "P", "format": P_format})
    writer.sheets["Schedule"].conditional_format("B2:AA105", {"type": "text", "criteria": "begins with",
                                                              "value": "N", "format": N_format})
    writer.sheets["Schedule"].conditional_format("B2:AA105", {"type": "text", "criteria": "ends with",
                                                              "value": "(S)", "format": S_format})
    # resize columns
    writer.sheets["Schedule"].set_column(0, 0, 25)
    writer.sheets["Schedule"].set_column(1, 28, 4)
    writer.save()

    print("Opening Excel File")
    import webbrowser
    webbrowser.open(output_file_path, "schedule")


if __name__ == '__main__':
    main()