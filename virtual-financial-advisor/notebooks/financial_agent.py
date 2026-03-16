# %%
from typing import TypedDict, Optional, List, Dict
import pandas as pd
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END
import streamlit as st

class FinancialAdvisorState(TypedDict):

    # User input
    data_path: str
    user_id: str
    user_query: str
    PER_API_KEY: str
    deployment_name: str
    
    time_period: Optional[str]

    # Financial analysis
    financial_summary: Optional[Dict]
    risk_flags: Optional[List[str]]
    simulation_results: Optional[Dict]

    # Agent control
    next_step: Optional[str]
    iteration_count: int
    confidence_score: Optional[float]

    # Final output
    personalized_advice: Optional[str]
    
    # Optional advanced features
    clarification_needed: Optional[bool]
    clarification_question: Optional[str]

def analyze_finances(state):

    df = pd.read_csv(state["data_path"])

    # Filter for the specific user
    df = df[df["user_id"] == state["user_id"]]

    # Total income (amount > 0)
    income = df[df["amount"] > 0]["amount"].sum()

    # Total expenses (amount < 0)
    expenses = abs(df[df["amount"] < 0]["amount"].sum())

    savings = income - expenses
    savings_rate = savings / income if income > 0 else 0

    # Category-wise expense breakdown
    category_spend_df = (
        df[df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )

    category_spend = category_spend_df.to_dict()

    summary = {
        "total_income": income,
        "total_expenses": expenses,
        "net_savings": savings,
        "savings_rate": round(savings_rate, 2),
        "top_expense_categories": category_spend
    }

    state["financial_summary"] = summary
    state["iteration_count"] = state["iteration_count"] + 1

    # print(state)
    return {
        "financial_summary": summary,
        "iteration_count": state["iteration_count"] + 1
        # state
    }

# %%
def detect_risks(state: FinancialAdvisorState):
    summary = state["financial_summary"]
    risks = []

    if summary["savings_rate"] < 0.15:
        risks.append("Low savings rate (<15%)")

    top_categories = summary["top_expense_categories"]
    
    if "Dining" in top_categories and top_categories["Dining"] > 0.2 * summary["total_expenses"]:
        risks.append("High dining expenses")

    if summary["net_savings"] < 0:
        risks.append("Spending exceeds income")

    # Simple emergency fund rule
    if summary["net_savings"] < summary["total_expenses"] * 3:
        risks.append("Emergency fund likely insufficient")

    state["risk_flags"] = risks
    state["iteration_count"] = state["iteration_count"] + 1

    return {
        "risk_flags": risks,
        "iteration_count": state["iteration_count"] + 1
    }

# %%
def simulate_scenarios(state: FinancialAdvisorState):
    summary = state["financial_summary"]
    simulations = {}

    income = summary["total_income"]
    expenses = summary["total_expenses"]

    # Scenario 1: Save 10% more
    new_savings = summary["net_savings"] + 0.1 * income
    simulations["increase_savings_10_percent"] = {
        "new_savings": new_savings,
        "new_savings_rate": round(new_savings / income, 2)
    }

    # Scenario 2: Reduce top expense by 15%
    top_category = next(iter(summary["top_expense_categories"]))
    reduction = 0.15 * summary["top_expense_categories"][top_category]

    new_expense = expenses - reduction
    new_savings = income - new_expense

    simulations["reduce_top_category_15_percent"] = {
        "category": top_category,
        "new_savings": new_savings,
        "new_savings_rate": round(new_savings / income, 2)
    }

    state["simulation_results"] = simulations
    state["iteration_count"] = state["iteration_count"] + 1

    return {
        "simulation_results": simulations,
        "iteration_count": state["iteration_count"] + 1
    }


# %%
def generate_advice(state: FinancialAdvisorState):

    prompt = f"""
    User Query: {state['user_query']}

    Financial Summary:
    {state['financial_summary']}

    Risk Flags:
    {state['risk_flags']}

    Simulation Results:
    {state['simulation_results']}

    Provide clear, personalized financial advice.
    Be specific and practical.
    """

    client = AzureOpenAI(
    api_key = state['PER_API_KEY'], # Put your DIAL API Key here
    api_version = "2024-02-01",
    azure_endpoint = "https://ai-proxy.lab.epam.com"
    )

    response = client.chat.completions.create(
    model= state['deployment_name'],
    temperature=0,
    messages=[
    {
    "role": "user",
    "content": prompt,
    },
    ],
    )

    advice = response.choices[0].message.content

    state["personalized_advice"] = advice,
    state["iteration_count"] = state["iteration_count"] + 1

    return {
    "personalized_advice": advice,
    "iteration_count": state["iteration_count"] + 1
    }

    # print("advice :", advice)
# %%
def reflect_and_score(state: FinancialAdvisorState):
    advice = state["personalized_advice"]

    if advice and len(advice) > 100:
        confidence = 0.9
        # print("length of advice is greater than 100")
    else:
        confidence = 0.5
        # print("length of advice is less than 100")

    state["confidence_score"] = confidence
    state["iteration_count"] = state["iteration_count"] + 1

    return {
        "confidence_score": confidence,
        "iteration_count": state["iteration_count"] + 1
    }   

    # print("confidence_score :", state["confidence_score"])

# %%
MAX_ITERATIONS = 10
CONFIDENCE_THRESHOLD = 0.8

def planner_node(state: FinancialAdvisorState):
    print("planning node: recieved, ", state)
    # Safety guard
    if state["iteration_count"] >= MAX_ITERATIONS:
        return {"next_step": "END"}

    # Step 1: Ensure financial analysis exists
    if state.get("financial_summary") is None:
        return {"next_step": "analyze_finances"}

    # Step 2: Ensure risk detection exists
    if state.get("risk_flags") is None:
        return {"next_step": "detect_risks"}

    # Step 3: If user query mentions simulation
    if "save" in state["user_query"].lower() or "what if" in state["user_query"].lower():
        if state.get("simulation_results") is None:
            return {"next_step": "simulate_scenarios"}

    # Step 4: Generate advice if not generated
    if state.get("personalized_advice") is None:
        return {"next_step": "generate_advice"}

    # Step 5: Reflect and score
    if state.get("confidence_score") is None:
        return {"next_step": "reflect_and_score"}

    # Step 6: If low confidence → regenerate advice
    if state["confidence_score"] < 0.8:
        return {"next_step": "generate_advice"}

    # Step 7: All done
    return {"next_step": "END"}


workflow = StateGraph(FinancialAdvisorState)

workflow.add_node("planner", planner_node)
workflow.add_node("analyze_finances", analyze_finances)
workflow.add_node("detect_risks", detect_risks)
workflow.add_node("simulate_scenarios", simulate_scenarios)
workflow.add_node("generate_advice", generate_advice)
workflow.add_node("reflect_and_score", reflect_and_score)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    lambda state: state["next_step"],
    {
        "analyze_finances": "analyze_finances",
        "detect_risks": "detect_risks",
        "simulate_scenarios": "simulate_scenarios",
        "generate_advice": "generate_advice",
        "reflect_and_score": "reflect_and_score",
        "END": END
    }
)

workflow.add_edge("analyze_finances", "planner")
workflow.add_edge("detect_risks", "planner")
workflow.add_edge("simulate_scenarios", "planner")
workflow.add_edge("generate_advice", "planner")
workflow.add_edge("reflect_and_score", "planner")

agent = workflow.compile()

st.write("This is an AI-powered virtual financial advisor that will help you with finanacial advices. Please provide your details")

with st.form('advisor'):
    str_user_id = st.text_input("Enter your user_id")
    str_query = st.text_input("Enter your query")
    str_PER_API_KEY = st.text_input("Enter the api key")
    str_deployment_name = st.text_input("Enter the deployment name")
    ts = st.form_submit_button("Generate advice")

if ts:

    initial_state = {
        "data_path": "C://Users//Suchi_Kumari//GenAI_Capston//virtual-financial-advisor//data//virtual_financial_advisor_data.csv",
        "user_id": str_user_id,
        "user_query": str_query,
        "PER_API_KEY": str_PER_API_KEY,
        "deployment_name": str_deployment_name,

        "financial_summary": None,
        "risk_flags": None,
        "simulation_results": None,
        "financial_goal": None,

        "personalized_advice": None,
        "confidence_score": None,

        "iteration_count": 0,
        "next_step": None
    }

    
    final_state = agent.invoke(initial_state)
    result = final_state["personalized_advice"]
    st.success(result)
