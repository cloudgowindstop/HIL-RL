from xtele.core.integrate_module import TeleCore
import time

m_tele = TeleCore()

print(m_tele.act())
m_tele.switch_reverse()

while True:
    try:
        m_tele.sync_position(
            [
                0.0,
                0.0,
                0.0,
                -1.5707963267948966,
                0.0,
                1.5707963267948966,
                0,
                0,
            ]
        )
        time.sleep(0.1)
    except KeyboardInterrupt:
        m_tele.tele_agent.close()
        exit(0)
