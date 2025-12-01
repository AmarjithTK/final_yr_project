Here is the comprehensive list of National and Regional holidays in Kerala, specifically structured for your dataset generation.

To make your Python script realistic, I have categorized these into **Fixed Dates** (easy to code) and **Variable Dates** (Lunar/Malayalam calendar based, which change every year).

### **1. Major Kerala Holidays (High Impact on Load)**
*These cause significant shifts in the load curve (e.g., Commercial load drops to near zero, Residential spikes).*

| Holiday Name | Type | Approx. Date | Load Impact Profile |
| :--- | :--- | :--- | :--- |
| **Republic Day** | National | **Jan 26** | Commercial/Industrial Off |
| **Vishu** | Regional (Malayalam New Year) | **Apr 14 or 15** | High Residential (Daytime Feast) |
| **May Day** | International | **May 01** | Industrial Strictly Off |
| **Independence Day** | National | **Aug 15** | Commercial/Industrial Off |
| **Onam Week** | Regional (Harvest Festival) | **Aug/Sept (Variable)** | **Massive Impact.** 4-5 days of high residential load, shops closed intermittently. |
| **Gandhi Jayanthi** | National | **Oct 02** | Commercial/Industrial Off |
| **Christmas** | Religious/Global | **Dec 25** | High Evening Residential Load (Lighting) |

---

### **2. Variable Religious & Regional Holidays**
*These depend on the lunar or Malayalam calendar. For your synthetic dataset, you can hardcode the dates for specific years (e.g., 2024/2025).*

**A. Hindu Festivals (Malayalam Calendar)**
* **Maha Shivaratri:** Feb/March (Night load spike due to sleepless vigil/temples).
* **Sree Narayana Guru Jayanthi:** Aug/Sept.
* **Sree Narayana Guru Samadhi:** Sept/Oct.
* **Mahanavami & Vijayadashami (Pooja Holidays):** Sept/Oct (2 days). *Industrial machinery often shut down for maintenance/worship.*
* **Deepavali:** Oct/Nov (Evening Lighting spike, though less intense in Kerala than North India).

**B. Muslim Festivals (Hijri Calendar)**
* **Eid-ul-Fitr (Ramadan end):** Apr/May.
* **Bakrid (Eid-al-Adha):** June/July.
* **Milad-i-Sherif:** Sept/Oct.
* *Note: These dates shift backward by ~11 days every year.*

**C. Christian Festivals**
* **Maundy Thursday & Good Friday:** Mar/Apr (Commercial load drops).
* **Easter:** Sunday following Good Friday.

---

### **3. Dates for Your Python Script (2024 Reference)**
You can use these specific dates to hardcode the `is_holiday` column in your generator function.

* **Jan 26:** Republic Day
* **Mar 08:** Maha Shivaratri
* **Mar 29:** Good Friday
* **Apr 10:** Eid-ul-Fitr
* **Apr 14:** Vishu
* **May 01:** May Day
* **Jun 17:** Bakrid
* **Aug 15:** Independence Day
* **Sep 14 - Sep 17:** **Onam Festival (Thiruvonam is Sep 15)** $\leftarrow$ *Mark this whole block as holiday.*
* **Oct 02:** Gandhi Jayanthi
* **Oct 12-13:** Pooja Holidays (Mahanavami/Vijayadashami)
* **Oct 31:** Deepavali
* **Dec 25:** Christmas

---

### **4. Seasonal Context for `generate_data` Logic**
When coding the "Kerala Weather" influence, use these specific windows:

1.  **Peak Summer (Mar 1 - May 31):**
    * **Temp:** $30^{\circ}\text{C} - 38^{\circ}\text{C}$.
    * **Load:** Max AC usage. Highest system peak.

2.  **South-West Monsoon (Jun 1 - Aug 31):**
    * **Weather:** Heavy Rain, High Humidity, Lower Temps ($24^{\circ}\text{C} - 29^{\circ}\text{C}$).
    * **Load:** AC load drops to near zero. Lighting load increases during daytime (dark clouds).

3.  **North-East Monsoon (Oct 1 - Nov 15):**
    * **Weather:** Afternoon Thunderstorms.
    * **Load:** Erratic. Frequent voltage dips/interruptions (can be simulated as noise).

4.  **Winter/Sabarimala Season (Nov 15 - Jan 15):**
    * **Weather:** Pleasant/Cool ($22^{\circ}\text{C} - 30^{\circ}\text{C}$).
    * **Load:** Early morning spike (4:00 AM) in residential areas (temple visits/water heating).

Would you like me to incorporate these specific dates into the Python generator script now?


The `load_type` column in this context is a **categorical label** (text) that tells your AI **"What is happening right now?"**

In a real distribution network, you usually just see the raw numbers (Power, Voltage). However, since we are creating **Synthetic Data** for training an LSTM, adding this column gives your model "context" to learn faster.

Here is exactly what that column should contain and why it matters:

### 1. What values will be in this column?
Instead of just "Residential" or "Industrial" (since a distribution transformer powers all of them simultaneously), this column should describe the **Dominant Regime** or **Day Type**.

The Python script will generate one of these values for every 15-minute row:

* **`Workday_Normal`**: Standard Mon-Sat behavior. Offices are open, factories are running.
* **`Weekend_Drop`**: Sundays. Commercial shops are closed, factories are on low shifts. Load profile changes shape.
* **`Holiday_Onam`**: A specific tag for the Onam season. This tells the LSTM: *"Ignore the office load, expect a massive spike in residential lighting and cooking load all day."*
* **`Holiday_General`**: Other holidays (Vishu, Christmas, Independence Day).
* **`Monsoon_Event`**: Days where heavy rain is simulated. This tells the LSTM: *"Expect temperature to drop and AC load to vanish."*

### 2. Why does your LSTM need it?
Without this column, the LSTM sees a drop in power on a Wednesday and thinks, "This is an error."
**With** this column, the LSTM sees the drop, looks at `load_type = "Holiday_Onam"`, and learns: *"Ah, it's a holiday on a Wednesday, so the drop is normal."*

### 3. How we will generate it (Logic Plan)
When I generate the code for you, I will use a logic map like this:

| Timestamp Condition | Assigned `load_type` | Effect on Power (P) |
| :--- | :--- | :--- |
| **Monday - Saturday** | `Workday` | Base Load = 100% |
| **Sunday** | `Weekend` | Base Load = 70% (Shops/Factories closed) |
| **Aug 29 - Sept 8** | `Festival_Onam` | Commercial = 20%, Residential = 150% |
| **June - July (Random)** | `Monsoon_Heavy` | AC Load = 0%, Lighting = 120% (Dark clouds) |

### 4. Important Note for your Model
When you feed this CSV into your LSTM later:
1.  **Input:** You will convert this text column into numbers (One-Hot Encoding or Embeddings).
2.  **Output:** Your model will use it to switch its internal logic based on whether it's a workday or a festival.

**Shall I proceed to generate the Python script with this logic included?**


This is a comprehensive architectural plan for generating a **Synthetic Kerala Distribution Network Dataset**. You can copy-paste the specific instructions below into a coding AI (like me or ChatGPT) to generate the Python script.

### **The Master Data Generation Plan**

**Objective:** Create a CSV dataset with 15-minute intervals containing electrical parameters ($P, Q, V, I, PF, KWH$) for a mixed-load distribution network in Kerala, India.

---

### **1. Input Parameters**
* **Time Resolution:** 15 minutes (`freq='15T'`).
* **Duration:** Variable `N_MONTHS` (default 12 or 24).
* **Nominal Voltage:** 11kV (Distribution level) or 415V (Low Tension). *Recommendation: 415V (3-phase) per node aggregate.*
* **Base Load Capacity:** 500 kW (Total Transformer Capacity).

---

### **2. The "Kerala" Environmental Model**
We need to mathematically simulate Kerala's specific climate behavior, as it drives air conditioning and fan loads.

* **Season Logic (Month-based):**
    * **Summer (March - May):** High Temperature ($30^{\circ}\text{C} - 38^{\circ}\text{C}$). **Peak Load Season.**
    * **Monsoon 1 (June - Aug):** South-West Monsoon. Temp drops ($24^{\circ}\text{C} - 29^{\circ}\text{C}$), high humidity. Load drops due to rain, but lighting load increases (dark clouds).
    * **Festive/Harvest (Aug - Sept):** **Onam Season**. Moderate temps.
    * **Monsoon 2 (Oct - Nov):** North-East Monsoon. Similar to SW Monsoon but shorter.
    * **Winter (Dec - Feb):** "Kerala Winter" (mild). Pleasant ($22^{\circ}\text{C} - 32^{\circ}\text{C}$). Lowest AC load.

* **Temperature Generation Formula:**
    $$T_{t} = T_{\text{base}} + T_{\text{seasonal}} + T_{\text{daily}} + \text{noise}$$
    * Use a sinusoidal wave peaking at 2 PM and lowest at 4 AM.
    * Apply a "Rain Penalty": Randomly drop temperature by $3^{\circ}\text{C}-5^{\circ}\text{C}$ during Monsoon months.

---

### **3. Load Profiling (The "Mix")**
A distribution network is never just one thing. We will model three distinct components and sum them up using a weighted average.

**A. Residential Load (40% Weight):**
* **Profile:** "Duck Curve" influenced.
* **Morning Peak (6 AM - 9 AM):** Water heaters, kitchen appliances.
* **Day Dip (10 AM - 5 PM):** People at work/school.
* **Evening Peak (6 PM - 10 PM):** Lighting, TV, ACs (Highest peak).
* **Night:** Low baseload (fans/AC cycling).

**B. Commercial Load (30% Weight):**
* **Profile:** "Box Curve".
* **9 AM - 6 PM:** High, steady load (Shops, Offices).
* **6 PM - 9 AM:** Near zero (Security lights/Servers only).
* **Weekends:** Drop this load by 70% on Sundays.

**C. Industrial Load (30% Weight):**
* **Profile:** "Flat Baseload" with spikes.
* **Shift Work:** Constant high load.
* **Power Factor:** Poor (Inductive loads like motors).

---

### **4. The "Kerala Calendar" (Events & Holidays)**
Hardcode specific dates to inject anomalies.

* **Sundays:** Reduction in Commercial/Industrial load.
* **Onam (Variable, usually Aug/Sept):**
    * *Effect:* Commercial load drops (shops close). Residential load Spikes significantly (feasts/lights) from 10 AM to 10 PM.
* **Vishu (April):** Similar to Onam but shorter duration.
* **Christmas/New Year (Dec):** Evening peaks increase (Decorative lighting).

---

### **5. Electrical Parameter Derivation (The Physics)**
Once we calculate the Total Active Power ($P_{\text{total}}$), we derive the rest using physics, not random numbers.

1.  **Active Power ($P$):**
    $$P_{\text{total}} = P_{\text{res}} + P_{\text{com}} + P_{\text{ind}} + P_{\text{weather\_factor}}$$
    *(Add Gaussian noise for realistic jitter).*

2.  **Power Factor ($PF$):**
    * $PF$ is dynamic, not static.
    * **Formula:** $PF_{t} = 0.95 - (0.1 \times \frac{P_{\text{ind}}}{P_{\text{total}}}) + \text{noise}$.
    * *Logic:* As Industrial load share increases, PF drops (more inductive). Residential is usually 0.9-0.95.

3.  **Reactive Power ($Q$):**
    * **Formula:** $Q = P \times \tan(\arccos(PF))$.
    * *(Result is in kVAR).*

4.  **Voltage ($V$):**
    * Inverse relationship with Load (Voltage Drop/Sag).
    * **Formula:** $V_{t} = V_{\text{nominal}} - (k \times P_{t}) + \text{random\_fluctuation}$.
    * *Logic:* High load causes voltage to sag. Low load (night) might see slight over-voltage.

5.  **Current ($I$):**
    * **Formula:** $I = \frac{\sqrt{P^2 + Q^2}}{\sqrt{3} \times V}$.
    * *(Assuming 3-phase).*

6.  **Energy ($KWH$):**
    * This must be cumulative (like a meter reading).
    * **Formula:** $KWH_{t} = KWH_{t-1} + (P_{t} \times 0.25)$.
    * *(0.25 because 15 mins is 1/4th of an hour).*

---

### **6. Prompt for the AI Coder**
*Copy and paste the block below to generate the exact Python code you need.*

***

**PROMPT START**

Write a Python script to generate a synthetic dataset for a Kerala, India based electrical distribution network.
**Requirements:**
1.  **Output:** CSV file named `kerala_distribution_data.csv`.
2.  **Columns:** `timestamp`, `temperature`, `is_holiday`, `load_type`, `P` (kW), `Q` (kVAR), `V` (Volts), `I` (Amps), `PF`, `KWH` (Cumulative).
3.  **Time:** 15-minute intervals (`freq='15T'`).
4.  **Function:** Create a function `generate_kerala_data(months=12)` that generates data for the specified number of months starting from Jan 1st, 2023.

**Logic Specifications:**
* **Load Composition:** Create 3 synthetic curves: Residential (Morning/Evening peaks), Commercial (9-5 block), Industrial (Flat). Combine them with weights [0.4, 0.3, 0.3].
* **Seasonality (Kerala Context):**
    * *Mar-May:* Multiply base load by 1.3 (Summer AC load).
    * *Jun-Aug:* Multiply by 0.9 (Monsoon cooling).
    * *Aug-Sep:* Randomly inject "Onam" spikes (high residential load).
* **Weekly Cycle:** On Sundays, drop Commercial/Industrial load by 50%.
* **Physics Formulas:**
    * Generate `P` first based on load curves + noise.
    * Calculate `PF` dynamically: varying between 0.85 (high load) and 0.98 (low load).
    * Calculate `Q` using $Q = P \times \tan(\arccos(PF))$.
    * Calculate `V` assuming voltage sag: $415 - (0.05 \times P) + \text{noise}$.
    * Calculate `I` using $I = \frac{\sqrt{P^2 + Q^2}}{\sqrt{3} \times V}$.
    * Calculate `KWH` as a cumulative sum of $(P \times 0.25)$.

**Libraries:** Use `pandas`, `numpy`. Ensure the data looks realistic with Gaussian noise added to smooth curves.

**PROMPT END**


This is the architectural blueprint for the **Kerala Environmental Model**. This document bridges the gap between Meteorology and Electrical Load Profiling.

Since you are generating synthetic data, we don't need exact historical weather; we need **statistically representative behavior** that drives air conditioners, fans, and lighting loads in a way that is unique to Kerala.

---

### **1. The Four Climatic Seasons of Kerala**

For the Python script, divide the year into these four distinct blocks. Each has different base parameters.

#### **A. Peak Summer (March 1 – May 31)**
* **The Vibe:** Scorching heat + High Humidity. "Sticky" weather.
* **Electrical Impact:**
    * **Max Load:** ACs running 24/7 (even at night).
    * **Fans:** Full speed.
    * **Pumps:** High usage (groundwater levels drop).
* **Data Ranges:**
    * *Temp ($T$):* Min $26^{\circ}\text{C}$ (Night) / Max $36^{\circ}\text{C} - 39^{\circ}\text{C}$ (Day).
    * *Humidity ($RH$):* $65\% - 85\%$.
    * *Anomaly:* **"Mango Showers"** (April/May). Sudden afternoon thunderstorms that drop temp by $5^{\circ}\text{C}$ for 2 hours.

#### **B. South-West Monsoon (June 1 – August 31)**
* **The Vibe:** The *Edavappathi*. Continuous, heavy rainfall. Dark days.
* **Electrical Impact:**
    * **AC Load:** Crashes (Ambient temp is cool).
    * **Lighting Load:** **Spikes during the day** (thick cloud cover blocks sun).
    * **Heating:** Water heater usage increases (bathing water feels cold).
* **Data Ranges:**
    * *Temp ($T$):* Min $23^{\circ}\text{C}$ / Max $29^{\circ}\text{C}$ (Very narrow range).
    * *Humidity ($RH$):* $90\% - 100\%$ (Saturation).
    * *Rain Factor:* High probability of "Rain Events" (Power interruptions typically happen here).

#### **C. Wet Transition / NE Monsoon (Sept 1 – Nov 15)**
* **The Vibe:** The *Thulavarsham*. Hot mornings, explosive thunderstorms in the afternoon/evening.
* **Electrical Impact:**
    * **Erratic Load:** ACs on in the morning, off in the evening.
    * **Voltage:** High fluctuation (lightning/transients).
* **Data Ranges:**
    * *Temp ($T$):* Min $24^{\circ}\text{C}$ / Max $32^{\circ}\text{C}$.
    * *Humidity ($RH$):* $75\% - 90\%$.

#### **D. "Winter" / Dry Season (Nov 15 – Feb 28)**
* **The Vibe:** Pleasant. Cool nights, bright sunny days.
* **Electrical Impact:**
    * **Min Load:** Windows often open; fans on low speed. AC usage minimal.
    * **Morning Peak:** 5 AM - 7 AM (Sabarimala season – early baths/water heating).
* **Data Ranges:**
    * *Temp ($T$):* Min $21^{\circ}\text{C}$ / Max $33^{\circ}\text{C}$.
    * *Humidity ($RH$):* $50\% - 70\%$ (Driest time of year).

---

### **2. Variable Generation Logic (The Math)**

To generate realistic 15-minute data, use these formulas in your Python script.

#### **Variable 1: Temperature ($T$)**
Kerala's temperature curve is not a perfect sine wave. It rises fast and cools slow.
$$T_t = T_{\text{base}} + A_{\text{day}} \cdot \sin(\text{time\_factor}) - \text{Rain\_Cooling} + \text{Noise}$$

* **Daily Cycle:** Lowest at 05:00 AM. Highest at 02:30 PM.
* **The "Tropical Night" Effect:** Unlike deserts, Kerala nights stay warm ($>25^{\circ}\text{C}$ in summer) due to humidity. *Do not let the night temp drop too low in your code.*

#### **Variable 2: Relative Humidity ($RH$)**
Humidity is inversely proportional to Temperature, but in Kerala, it creates the **Heat Index**.
* **Logic:** When $T$ is high, $RH$ is lowest ($60\%$). When $T$ drops (night/rain), $RH$ spikes ($95\%$).

#### **Variable 3: The Heat Index (Apparent Temperature)**
*Crucial for Load Prediction.*
People turn on ACs based on how hot it *feels*, not the thermometer.
**Formula:**
$$HI \approx T + 0.55 \times (T - 25)$$
*(Simplified version for coding. Use this to drive the Residential Load curve).*

#### **Variable 4: Solar Irradiance / Cloud Cover**
* **Clear Sky:** Jan - Apr (High solar generation if you add solar panels later).
* **Overcast:** Jun - Jul (Near zero solar generation, High lighting load).

---

### **3. The "Kerala Weather Matrix" (Copy-Paste for Code)**

Use this dictionary in your Python script to set boundaries for the random generator.

| Month | Season Label | Temp Range ($^{\circ}\text{C}$) | Humidity Range (%) | Rain Prob. | Load Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Jan** | Winter | 22 - 32 | 50 - 75 | 5% | Low / Moderate |
| **Feb** | Winter/Summer | 23 - 34 | 55 - 80 | 10% | Rising Trend |
| **Mar** | Summer | 25 - 36 | 60 - 85 | 15% | High (ACs start) |
| **Apr** | Summer (Peak) | 27 - 38 | 65 - 90 | 30% | **Max Peak** |
| **May** | Summer (Peak) | 26 - 36 | 70 - 90 | 50% | High (Mango showers) |
| **Jun** | SW Monsoon | 24 - 29 | 85 - 100 | **90%** | Low (AC Off, Lights On) |
| **Jul** | SW Monsoon | 23 - 28 | 90 - 100 | **95%** | **Minimum AC Load** |
| **Aug** | Monsoon/Onam | 24 - 30 | 85 - 95 | 70% | Moderate |
| **Sep** | Wet Transition | 24 - 31 | 80 - 90 | 60% | Variable |
| **Oct** | NE Monsoon | 24 - 31 | 80 - 95 | 60% | Erratic (Thunderstorms) |
| **Nov** | NE/Winter | 23 - 31 | 70 - 85 | 40% | Moderate |
| **Dec** | Winter | 21 - 31 | 60 - 75 | 10% | Low |

---

### **4. Specific Weather Events to Simulate**

1.  **The "Rain Cooling" Function:**
    * If `is_raining == True`: Drop Temperature by $4^{\circ}\text{C}$ over 30 minutes.
    * *Effect on Grid:* Instant drop in load (ACs turn off).

2.  **The "Dark Day" Function:**
    * During June/July, set `is_overcast = True` from 9 AM to 4 PM.
    * *Effect on Grid:* Residential lighting load increases by 20% compared to a sunny day.

3.  **Humidity Stress:**
    * If Temp $> 32^{\circ}\text{C}$ AND Humidity $> 80\%$: Apply a "Discomfort Multiplier" of $1.2x$ to the Residential Load. (This simulates ACs working harder and people running them at lower setpoints).

---

### **Summary for the AI Coder**
When you generate the code, ask for a function `get_weather_params(month, hour)` that:
1.  Returns `temp`, `humidity`, `rain_status`.
2.  Interpolates values smoothly between min/max based on the hour.
3.  Adds Gaussian noise.
4.  Applies the "Rain Penalty" to temperature if it's a monsoon month.

This will give you a dataset where the **Power ($P$)** actually correlates with the **Weather**, making your LSTM training significantly more effective.


Yes, that is correct for an 11 kV system.

Here is the context for why it peaks at 133 A:

High Voltage means Lower Current: Because the voltage is high (11,000 V), you can deliver a lot of power with less current.
133 Amps at 11 kV is roughly 2.5 Megawatts (MW) of power.
If this were a standard home line (230V), that same amount of power would require over 10,000 Amps!
Feeder Capacity: You set the BASE_MVA = 5.0 (5 MVA Capacity).
The maximum current this feeder can handle is approx 262 Amps.
So at 133 A, your feeder is running at about 50% load, which is a very healthy, realistic operating level.
Do you want to simulate a heavier load? If you want to push the current closer to the limit (e.g., 200-250 A), we can simply double the NUM_CONSUMERS to 2400. Let me know if you want me to do that.