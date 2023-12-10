
# Tabular Q-Learning Controller

import numpy as np
import pandas as pd
import simulation_environment as sim_env
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pi_controller import PIController, apply_system_constraints
import pytz

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

external_temperatures = pd.read_csv(r'./input_data/External_temperatures_May_2022-March_2023.csv') # May 2022 to March 2023 data

# Setpoints (accounting for time zone change in spreadsheet) 
daily_temperature_sp = [16]*7 + [19]*17
number_of_repetitions_UTC = (len(external_temperatures) - (182*24 + 1))// len(daily_temperature_sp)
external_temperatures['temperature_sp'] = ([16]*7+[19]*17)*182 + [16] + daily_temperature_sp * number_of_repetitions_UTC + daily_temperature_sp[:((len(external_temperatures) - (182*24 + 1)) % len(daily_temperature_sp))]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

number_of_states= 21  # Number of states 
number_of_actions = 11  # Number of actions 
learning_rate = 0.001 # alpha
discount_factor = 0.99 # gamma
exploration_probability  = 0.4 # epsilon

# system constraints
rc_vals = [0.0496, 0.5862, 0.1383, 59.01, 247.8, 0.1576, 0.0289, 1749100, 388300]
sales_area_dims = [117, 64.866, 9]  # width, length, height
q_max = 500000  # Maximum possible thermal energy output from the HVAC system (W)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Initialise Q-table
Q = np.zeros((number_of_states, number_of_actions))


# Discretise the states
def calculate_state_index(setpoint_error):
    negative_deviation_max = -10
    positive_deviation_max = 10
    interval = (positive_deviation_max - negative_deviation_max) / (number_of_states-1)
    if interval < negative_deviation_max:
        state_index = 0
    elif interval > positive_deviation_max:
        state_index = number_of_states-1
    else:
        state_index = int((setpoint_error - negative_deviation_max)/interval)
    return state_index


# Reward function
def calculate_reward(setpoint_error, q_hvac):
    if setpoint_error > 0:
        temp_reward = 0
        energy_reward = q_hvac / 500000
    elif setpoint_error <-2:
        temp_reward =-abs(setpoint_error)**3
        energy_reward = 5*(q_hvac / 500000)
    else:
        temp_reward = -2*abs(setpoint_error)
        energy_reward = (q_hvac / 500000)
    total_reward = temp_reward - energy_reward
    return total_reward


# Calculate HVAC heat load
def calculate_q_hvac(action_index, previous_q_hvac):
    
    delta_q_hvac_range = 500000 
    # Calculate potential state change based on delta_q, where delta_q ranges between -250k and +250k W
    delta_q = delta_q_hvac_range * action_index /(number_of_actions-1) - (delta_q_hvac_range/2)
    potential_q_hvac = previous_q_hvac + delta_q

    # Check if potential_q_hvac exceeds constraints
    if potential_q_hvac > q_max:
        delta_q = q_max - previous_q_hvac

    if potential_q_hvac < 0:
        delta_q = - previous_q_hvac
        
    q_hvac = previous_q_hvac + delta_q
    return q_hvac

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Training the controller
number_of_runs = 20 # Number of times to run over the 11 months data for training 
for run in range(number_of_runs):   
    
    total_reward = 0
    q_hvac = 0
    T_init = [18]*13 # Intitial temperature recorded by all 16 temperature sensors, assuming uniform distribution
    tempin = T_init[-1]
    setpoint_error =  tempin - external_temperatures['temperature_sp'][0]
    state_index = calculate_state_index(setpoint_error)
    thermal_model = sim_env.ThermalNetwork(rc_vals, sales_area_dims, T_init, dt=3600)
    
    for T in range(len(external_temperatures)):
        # Exploration vs. exploitation
        if np.random.rand() < exploration_probability: # explore
            action_index = np.random.choice(number_of_actions)
        else: # greedy search
            action_index = np.argmax(Q[state_index])     
        q_hvac = calculate_q_hvac(action_index, q_hvac)
        model_inputs = np.array([external_temperatures['temperature_0'][T]] * 5 + [q_hvac])
        T_next_timestep = thermal_model.calcNextTimestep(model_inputs)
        tempin = T_next_timestep[-1]
        setpoint_error =  tempin - external_temperatures['temperature_sp'][T]
        state_index = calculate_state_index(setpoint_error)
        reward = calculate_reward(setpoint_error, q_hvac)
        total_reward += reward
        Q[state_index, action_index] = (1 - learning_rate) * Q[state_index, action_index] + learning_rate * (reward + discount_factor * np.max(Q[state_index]))
    
    print(f"Run {run+1}: Total Reward = {total_reward}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                             TESTING THE CONTROLLER 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


external_temperatures = pd.read_csv(r'./input_data/External_temperatures_May-July_2023.csv') # May to July 2023 data
external_temperatures['Timestamp'] = pd.to_datetime(external_temperatures['Timestamp']) # Convert to datetime format
external_temperatures.set_index('Timestamp', inplace=True) # Set the 'Timestamp' column as the index

# Setpoints
daily_temperature_sp = [16]*7 + [19]*17  
numer_of_repetitions = len(external_temperatures) // len(daily_temperature_sp)
external_temperatures['temperature_sp'] = daily_temperature_sp * numer_of_repetitions + daily_temperature_sp[:len(external_temperatures) % len(daily_temperature_sp)]
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

crop1=0
crop2=336
external_temperatures = external_temperatures[crop1:crop2]
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

internal_temperatures = []
q_values = []

comfort_violation_Q_table = 0
total_thermal_energy_used_Q_table = 0
setpoint_error =  tempin - external_temperatures['temperature_sp'][0]
state_index = calculate_state_index(setpoint_error)
q_hvac=0
total_reward = 0
T_init = [18] * 13 
thermal_model = sim_env.ThermalNetwork(rc_vals, sales_area_dims, T_init, dt=3600)
tempin=T_init[-1]   
 
for T in range(len(external_temperatures)):
    action_index = np.argmax(Q[state_index,:])   
    internal_temperatures.append(tempin)

    if setpoint_error < 0 : # Comfort violation - Only account for deviations below setpoints
        comfort_violation_Q_table += abs(setpoint_error)*(24*365.25)/(crop2-crop1) # Kelvin hour per year 
    else:
        comfort_violation_Q_table += 0

    q_hvac = calculate_q_hvac(action_index, q_hvac)
    q_values.append(q_hvac/1000) # Convert to kW before appending to list 
    total_thermal_energy_used_Q_table += (q_hvac/1000)*(24*365.25)/(crop2-crop1) # kWh per year - divide q_hvac by 1000 since we use timesteps of 1 hour
    model_inputs = np.array([external_temperatures['temperature_0'][T]] * 5 + [q_hvac])
    T_next_timestep = thermal_model.calcNextTimestep(model_inputs)
    reward = calculate_reward(setpoint_error, q_hvac)
    total_reward += reward
    tempin = T_next_timestep[-1]
    setpoint_error =  tempin - external_temperatures['temperature_sp'][T]
    state_index=calculate_state_index(setpoint_error)
        
print(f"Total Reward = {total_reward}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PI controller (for comparison)
comfort_violation_PI = 0
total_thermal_energy_used_PI = 0
tempin = T_init[-1]
q_valuesPI=[]
internal_temperaturesPI=[]
pi_controller = PIController(max_heat_duty=q_max)
thermal_model = sim_env.ThermalNetwork(rc_vals, sales_area_dims, T_init, dt=3600)
for T in range(len(external_temperatures)):
    setpoint_error =  tempin - external_temperatures['temperature_sp'][T] 
    
    if setpoint_error < 0 : # Comfort violation - Only account for deviations below setpoints 
        comfort_violation_PI += abs(setpoint_error)*(24*365.25)/(crop2-crop1) # Kelvin hour per year
    else:
        comfort_violation_PI += 0

    pi_signal = pi_controller.pi_signal(external_temperatures['temperature_sp'][T], tempin) 
    q_hvac = apply_system_constraints(pi_signal, q_max) 
    q_valuesPI.append(q_hvac/1000) # Convert to kW before appending to list 
    total_thermal_energy_used_PI += (q_hvac/1000)*(24*365.25)/(crop2-crop1) # kWh per year
    model_inputs = np.array([external_temperatures['temperature_0'][T]] * 5 + [q_hvac])
    T_next_timestep = thermal_model.calcNextTimestep(model_inputs) 
    internal_temperaturesPI.append(tempin)
    tempin = T_next_timestep[-1]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

timesteps = external_temperatures.index

# Create a figure and a grid of subplots with one column and three rows
fig, axes = plt.subplots(3, figsize=(7,7), dpi=300, sharex=True)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator, tz=pytz.timezone('Europe/London'))

# Plot external temperature on the first subplot
axes[0].plot(timesteps, external_temperatures['temperature_0'], label='External Temperature', color='royalblue')
axes[0].xaxis.set_major_formatter(formatter)
axes[0].set_ylabel('[°C]')
axes[0].legend()

# Plot q_values on the second subplot 
axes[1].plot(timesteps, q_valuesPI, label='Thermal Power from HVAC System (PI controller)', color='royalblue')
axes[1].plot(timesteps, q_values, label='Thermal Power from HVAC System (Q-Table controller)', color='darkorange')
axes[1].xaxis.set_major_formatter(formatter)
axes[1].set_ylabel('[kW]')
axes[1].legend()

# Plot internal temperature and temperature setpoint on the third subplot
axes[2].plot(timesteps, external_temperatures['temperature_sp'], label='Temperature Setpoint', linestyle='--', color='black', linewidth=1)
axes[2].plot(timesteps, internal_temperaturesPI, label= 'Internal Store Temperature (PI controller)', color='royalblue')
axes[2].plot(timesteps, internal_temperatures, label='Internal Store Temperature (Q-Table controller)', color='darkorange')
axes[2].xaxis.set_major_formatter(formatter)
axes[2].set_ylabel('[°C]')
axes[2].legend()

plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#KPIs
electricity_price = 0.2 # £ per kWh_electrical
COP = 3 # kWh_thermal per kWh_electrical
CO2e_conversion_factor =  0.207074 # kg CO2e per kWh

total_electrical_energy_used_Q_table = total_thermal_energy_used_Q_table / COP # kWh_electrical
total_energy_cost_Q_table = total_electrical_energy_used_Q_table * electricity_price # £ per year
CO2e_emissions_Q_table = total_electrical_energy_used_Q_table * CO2e_conversion_factor  # kg CO2
print(" ")
print(f"Comfort Violation (Q Table) = {comfort_violation_Q_table} Kelvin hour / year")
print(f"Total Thermal Energy Used (Q Table)= {total_thermal_energy_used_Q_table} kWh/year")
print(f"Total Energy Cost (Q Table)= {total_energy_cost_Q_table} £/year")
print(f"CO2e Emissions (Q Table)= {CO2e_emissions_Q_table} kgCO2e/year")

total_electrical_energy_used_PI = total_thermal_energy_used_PI / COP # kWh_electrical
total_energy_cost_PI = total_electrical_energy_used_PI * electricity_price # £ per year
CO2e_emissions_PI = total_electrical_energy_used_PI * CO2e_conversion_factor # kg CO2
print(" ")
print(f"Comfort Violation (PI) = {comfort_violation_PI} Kelvin hour / year")
print(f"Total Thermal Energy Used (PI) = {total_thermal_energy_used_PI} kWh/year")
print(f"Total Energy Cost (PI)= {total_energy_cost_PI} £/year")
print(f"CO2e Emissions (PI)= {CO2e_emissions_PI} kgCO2e/year")






