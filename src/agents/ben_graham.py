from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import math
from i18n import get_text as _


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    # 使用翻译的docstring
    __doc__ = _("ben_graham.docstrings.agent")
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.fetching_metrics"))
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.gathering_line_items"))
        financial_line_items = search_line_items(ticker, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.getting_market_cap"))
        market_cap = get_market_cap(ticker, end_date)

        # Perform sub-analyses
        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.analyzing_earnings"))
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)

        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.analyzing_strength"))
        strength_analysis = analyze_financial_strength(financial_line_items)

        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.analyzing_valuation"))
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap)

        # Aggregate scoring
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # total possible from the three analysis functions

        # Map total_score to signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "earnings_analysis": earnings_analysis, "strength_analysis": strength_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.generating_analysis"))
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status("ben_graham_agent", ticker, _("ben_graham.status_messages.done"))

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # Store signals in the overall state
    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    # 使用翻译的docstring
    __doc__ = _("ben_graham.docstrings.analyze_earnings_stability")
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": score, "details": _("ben_graham.analysis_details.insufficient_data")}

    eps_vals = []
    for item in financial_line_items:
        if item.earnings_per_share is not None:
            eps_vals.append(item.earnings_per_share)

    if len(eps_vals) < 2:
        details.append(_("ben_graham.analysis_details.not_enough_eps"))
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append(_("ben_graham.analysis_details.eps_all_positive"))
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append(_("ben_graham.analysis_details.eps_mostly_positive"))
    else:
        details.append(_("ben_graham.analysis_details.eps_negative"))

    # 2. EPS growth from earliest to latest
    if eps_vals[0] > eps_vals[-1]:
        score += 1
        details.append(_("ben_graham.analysis_details.eps_grew"))
    else:
        details.append(_("ben_graham.analysis_details.eps_not_grew"))

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_strength(financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    # 使用翻译的docstring
    __doc__ = _("ben_graham.docstrings.analyze_financial_strength")
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": _("ben_graham.analysis_details.no_financial_data")}

    latest_item = financial_line_items[0]
    total_assets = latest_item.total_assets or 0
    total_liabilities = latest_item.total_liabilities or 0
    current_assets = latest_item.current_assets or 0
    current_liabilities = latest_item.current_liabilities or 0

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(_("ben_graham.analysis_details.current_ratio_solid").format(ratio=current_ratio))
        elif current_ratio >= 1.5:
            score += 1
            details.append(_("ben_graham.analysis_details.current_ratio_moderate").format(ratio=current_ratio))
        else:
            details.append(_("ben_graham.analysis_details.current_ratio_weak").format(ratio=current_ratio))
    else:
        details.append(_("ben_graham.analysis_details.current_ratio_missing"))

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(_("ben_graham.analysis_details.debt_ratio_conservative").format(ratio=debt_ratio))
        elif debt_ratio < 0.8:
            score += 1
            details.append(_("ben_graham.analysis_details.debt_ratio_acceptable").format(ratio=debt_ratio))
        else:
            details.append(_("ben_graham.analysis_details.debt_ratio_high").format(ratio=debt_ratio))
    else:
        details.append(_("ben_graham.analysis_details.debt_ratio_missing"))

    # 3. Dividend track record
    div_periods = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append(_("ben_graham.analysis_details.dividends_majority"))
            else:
                details.append(_("ben_graham.analysis_details.dividends_some"))
        else:
            details.append(_("ben_graham.analysis_details.dividends_none"))
    else:
        details.append(_("ben_graham.analysis_details.dividends_missing"))

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_graham(financial_line_items: list, market_cap: float) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    # 使用翻译的docstring
    __doc__ = _("ben_graham.docstrings.analyze_valuation_graham")
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": _("ben_graham.analysis_details.insufficient_valuation")}

    latest = financial_line_items[0]
    current_assets = latest.current_assets or 0
    total_liabilities = latest.total_liabilities or 0
    book_value_ps = latest.book_value_per_share or 0
    eps = latest.earnings_per_share or 0
    shares_outstanding = latest.outstanding_shares or 0

    details = []
    score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(_("ben_graham.analysis_details.ncav_value").format(value=net_current_asset_value))
        details.append(_("ben_graham.analysis_details.ncav_per_share").format(value=net_current_asset_value_per_share))
        details.append(_("ben_graham.analysis_details.price_per_share").format(value=price_per_share))

        if net_current_asset_value > market_cap:
            score += 4  # Very strong Graham signal
            details.append(_("ben_graham.analysis_details.net_net_strong"))
        else:
            # For partial net-net discount
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append(_("ben_graham.analysis_details.net_net_moderate"))
    else:
        details.append(_("ben_graham.analysis_details.net_net_missing"))

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(_("ben_graham.analysis_details.graham_number").format(value=graham_number))
    else:
        details.append(_("ben_graham.analysis_details.graham_number_missing"))

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(_("ben_graham.analysis_details.margin_of_safety").format(value=margin_of_safety))
            if margin_of_safety > 0.5:
                score += 3
                details.append(_("ben_graham.analysis_details.margin_high"))
            elif margin_of_safety > 0.2:
                score += 1
                details.append(_("ben_graham.analysis_details.margin_some"))
            else:
                details.append(_("ben_graham.analysis_details.margin_low"))
        else:
            details.append(_("ben_graham.analysis_details.price_invalid"))
    # else: already appended details for missing graham_number

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of Benjamin Graham:
    - Value emphasis, margin of safety, net-nets, conservative balance sheet, stable earnings.
    - Return the result in a JSON structure: { signal, confidence, reasoning }.
    """
    # 使用翻译的docstring
    __doc__ = _("ben_graham.docstrings.generate_graham_output")

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            _("ben_graham.prompt.system")
        ),
        (
            "human",
            _("ben_graham.prompt.human")
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning=_("ben_graham.analysis_details.default_error"))

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )
