import logging
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical triage assistant working alongside nurses at patient intake.

Your job:
1. Greet the patient and collect their presenting complaint and relevant history conversationally.
2. Gather enough structured data to run the triage screen (age, gender, vitals, comorbidities).
3. Once you have sufficient data, call run_triage_screen and present the result clearly.
4. Use search-drugs, search-clinical-guidelines, or search-medical-literature if the clinician
   asks about medications, treatment protocols, or relevant evidence.

Required fields before calling run_triage_screen:
- age, gender, hypertension (yes/no), heart_disease (yes/no)
- For stroke screen: ever_married, work_type, residence type, avg glucose, smoking status
- For diabetes screen: BMI, smoking history, HbA1c, blood glucose

Triage levels you will report: critical → urgent → semi-urgent → non-urgent.

Always include the disclaimer from the triage result. Never diagnose — you provide decision support.
If the patient describes emergency symptoms (chest pain, facial droop, severe dyspnoea),
escalate immediately and recommend calling emergency services before running any tool."""


class TriageAgent:
    def __init__(self):
        self._clinical_mcp: MCPServerStdio | None = None
        self._medical_mcp: MCPServerStdio | None = None
        self.agent: Agent | None = None

    async def _try_start_server(self, server: MCPServerStdio, name: str) -> MCPServerStdio | None:
        try:
            await server.__aenter__()
            logger.info("[triage_agent] %s MCP server started", name)
            return server
        except Exception as e:
            logger.warning("[triage_agent] %s MCP server failed to start: %s", name, e)
            return None

    async def start(self):
        clinical = MCPServerStdio(
            params={"command": "python", "args": ["-m", "clinical_mcp.server"]},
            cache_tools_list=True,
        )
        medical = MCPServerStdio(
            params={"command": "npx", "args": ["-y", "medical-mcp"]},
            cache_tools_list=True,
        )
        self._clinical_mcp = await self._try_start_server(clinical, "clinical")
        self._medical_mcp = await self._try_start_server(medical, "medical")

        active_servers = [s for s in [self._clinical_mcp, self._medical_mcp] if s is not None]
        self.agent = Agent(
            name="Triage Assistant",
            instructions=SYSTEM_PROMPT,
            mcp_servers=active_servers,
        )
        logger.info("[triage_agent] ready with %d MCP server(s)", len(active_servers))

    async def stop(self):
        if self._clinical_mcp:
            await self._clinical_mcp.__aexit__(None, None, None)
        if self._medical_mcp:
            await self._medical_mcp.__aexit__(None, None, None)

    async def chat(self, history: list[dict]) -> str:
        result = await Runner.run(self.agent, input=history)
        return result.final_output


triage_agent = TriageAgent()
