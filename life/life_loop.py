import time
from typing import Optional

from config import Config
from life.journal import Journal
from life.goals import GoalStore
from life.curiosity import Curiosity
from life.autolearn import AutoLearner
from tools.web_tools import WebTools, decode_ddg_url
from tools.read_tools import ReadTools


class LifeLoop:
    def __init__(self, codex):
        self.codex = codex
        self.journal = Journal()
        self.goals = GoalStore()
        self.curiosity = Curiosity(self.journal, self.goals)
        self.learner = AutoLearner(codex, self.journal)
        self.step = 0

    def _do_step(self) -> None:
        self.step += 1
        self.journal.add("thought", f"шаг {self.step}")

        self.curiosity.bootstrap(min_goals=3)
        self.curiosity.ensure_goal()

        goal = self.goals.pick_next()
        if goal is None:
            self.journal.add("thought", "нет целей — ставлю случайную")
            self.curiosity.bootstrap(min_goals=1)
            goal = self.goals.pick_next()
            if goal is None:
                self.journal.add("error", "не удалось поставить цель")
                return

        self.journal.add("goal", goal.text, source=goal.source)

        topic = self._extract_topic(goal.text)
        if topic is None or not self.learner.is_meaningful_question(topic):
            self.goals.complete(goal, result="тема не годится для поиска")
            self.journal.add("thought", f"пропускаю бессмысленную цель: {goal.text[:60]}")
            return

        question = topic
        self.journal.add("action", f"поиск: {question}")

        search_result = WebTools.search(question, limit=3)
        if search_result and "Ошибка" not in search_result and "ничего не нашёл" not in search_result:
            self.journal.add("result", search_result[:500], question=question)
            self.learner.learn_from_search_results(question, search_result)

            url = self._first_url(search_result)
            if url:
                self.journal.add("action", f"чтение страницы: {url}")
                page = ReadTools.read_url(url)
                if page and "Ошибка" not in page and "пуста" not in page:
                    self.journal.add("result", page[:500], url=url)
                    if self.learner.learn_from_page(question, page, source=url):
                        self.journal.add("candidate", f"ожидает проверки: {question[:60]}")
                else:
                    self.journal.add("error", f"чтение страницы: {page[:120]}")
        else:
            self.journal.add("error", f"поиск не удался: {search_result[:120]}")

        self.curiosity.from_unknown_words(goal.text)
        self.goals.complete(goal, result=f"обработано: {question}")
        self.journal.add("thought", f"цель завершена: {goal.text}")

        self.learner.maybe_train(self.step)

    @staticmethod
    def _extract_topic(goal_text: str) -> Optional[str]:
        text = goal_text.strip()
        for prefix in (
            "узнать что такое ",
            "узнать ",
            "поискать информацию про ",
            "поискать интересный факт о ",
            "поискать интересный факт об ",
            "поискать полезный совет по ",
            "поискать ",
            "научиться отвечать на ",
            "научиться ",
            "что такое ",
            "интересный факт о ",
            "интересный факт об ",
            "полезный совет по ",
        ):
            if text.lower().startswith(prefix):
                return text[len(prefix):].strip() or None
        return text or None

    @staticmethod
    def _first_url(search_text: str) -> Optional[str]:
        for line in search_text.splitlines():
            line = line.strip()
            if line.startswith("http://") or line.startswith("https://"):
                return decode_ddg_url(line)
        return None

    def run(self, max_steps: Optional[int] = None) -> None:
        max_steps = max_steps or Config.LIFE_MAX_STEPS_PER_SESSION
        print("=" * 60)
        print("KoKs: жизнь началась")
        print(f"Шагов за сессию: {max_steps}")
        print(f"Пауза между шагами: {Config.LIFE_STEP_PAUSE_SEC} c")
        print(f"Сон каждые {Config.LIFE_SLEEP_EVERY_STEPS} шагов на {Config.LIFE_SLEEP_SEC} c")
        print("=" * 60)

        self.journal.add("thought", "проснулся")

        try:
            for _ in range(max_steps):
                try:
                    self._do_step()
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    self.journal.add("error", f"шаг упал: {error}")
                time.sleep(Config.LIFE_STEP_PAUSE_SEC)

                if (
                    Config.LIFE_SLEEP_EVERY_STEPS > 0
                    and self.step % Config.LIFE_SLEEP_EVERY_STEPS == 0
                ):
                    self.journal.add("thought", f"сон на {Config.LIFE_SLEEP_SEC} c")
                    print(f"[сон {Config.LIFE_SLEEP_SEC} c]")
                    time.sleep(Config.LIFE_SLEEP_SEC)

        except KeyboardInterrupt:
            self.journal.add("thought", "прерван пользователем")
            print("\nЖизнь остановлена пользователем.")
        finally:
            self.journal.add("thought", "уснул")
            self.goals.save()
            stats = self.journal.stats()
            print("=" * 60)
            print("Статистика журнала:")
            for kind, count in stats.items():
                print(f"  {kind}: {count}")
            print("=" * 60)