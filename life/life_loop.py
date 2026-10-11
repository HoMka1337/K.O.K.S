
import time
from typing import Optional

from config import Config
from life.autolearn import AutoLearner
from life.curiosity import Curiosity
from life.goals import GoalStore
from life.journal import Journal
from tools.read_tools import ReadTools
from tools.web_tools import (
    WebTools,
    decode_ddg_url,
    is_public_url,
)


class LifeLoop:
    def __init__(self, codex):
        self.codex = codex
        self.journal = Journal()
        self.goals = GoalStore()
        self.curiosity = Curiosity(
            self.journal,
            self.goals
        )
        self.learner = AutoLearner(
            codex,
            self.journal
        )
        self.step = 0

    @staticmethod
    def _network_available() -> bool:
        return bool(
            getattr(Config, "ALLOW_NETWORK", False)
            and getattr(Config, "LIFE_ALLOW_NETWORK", False)
            and getattr(Config, "TOOLS_ENABLE_WEB", True)
        )

    @staticmethod
    def _search_succeeded(result: str) -> bool:
        if not isinstance(result, str):
            return False

        result = result.strip()

        if not result:
            return False

        blocked_starts = (
            "ошибка",
            "поиск ничего не нашёл",
            "сеть запрещена",
            "интернет-поиск отключён",
            "слишком много операций",
            "заблокировано",
            "укажи"
        )

        return not result.casefold().startswith(
            blocked_starts
        )

    @staticmethod
    def _page_succeeded(page: str) -> bool:
        if not isinstance(page, str):
            return False

        page = page.strip()

        if len(page) < 100:
            return False

        blocked_starts = (
            "ошибка",
            "страница пуста",
            "не удалось прочитать",
            "сеть запрещена",
            "чтение интернет-страниц отключено",
            "недопустимый",
            "неподдерживаемый",
            "слишком много операций",
            "заблокировано"
        )

        return not page.casefold().startswith(
            blocked_starts
        )

    def _do_step(self) -> None:
        self.step += 1

        self.journal.add(
            "thought",
            f"Шаг {self.step}"
        )

        self.curiosity.bootstrap(
            min_goals=3
        )

        self.curiosity.ensure_goal()

        goal = self.goals.pick_next()

        if goal is None:
            self.curiosity.bootstrap(
                min_goals=1
            )

            goal = self.goals.pick_next()

            if goal is None:
                self.journal.add(
                    "thought",
                    "Нет доступных целей"
                )
                return

        self.journal.add(
            "goal",
            goal.text,
            source=goal.source
        )

        topic = self._extract_topic(
            goal.text
        )

        if (
            not topic
            or not self.learner.is_meaningful_question(topic)
        ):
            self.goals.complete(
                goal,
                result="Тема не подходит для поиска"
            )

            self.journal.add(
                "thought",
                "Цель пропущена",
                goal=goal.text
            )
            return

        if not self._network_available():
            self.journal.add(
                "thought",
                "Поиск пропущен: сеть отключена",
                goal=goal.text
            )
            return

        question = topic

        self.journal.add(
            "action",
            f"Поиск: {question}"
        )

        try:
            search_result = WebTools.search(
                question,
                limit=3
            )

        except Exception as error:
            self.journal.add(
                "error",
                f"Ошибка поиска: {error}",
                question=question
            )
            return

        if not self._search_succeeded(search_result):
            self.journal.add(
                "error",
                f"Поиск не удался: {str(search_result)[:300]}",
                question=question
            )

            self.goals.complete(
                goal,
                result="Не удалось выполнить поиск"
            )
            return

        self.journal.add(
            "result",
            search_result[:500],
            question=question
        )

        self.learner.learn_from_search_results(
            question,
            search_result
        )

        url = self._first_url(
            search_result
        )

        saved = False

        if url:
            self.journal.add(
                "action",
                f"Чтение страницы: {url}",
                question=question
            )

            try:
                page = ReadTools.read_url(
                    url
                )

            except Exception as error:
                page = f"Ошибка чтения страницы: {error}"

            if self._page_succeeded(page):
                self.journal.add(
                    "result",
                    page[:500],
                    question=question,
                    source=url
                )

                saved = self.learner.learn_from_page(
                    question,
                    page,
                    source=url
                )

                if saved:
                    self.journal.add(
                        "thought",
                        "Материал ожидает проверки",
                        question=question,
                        source=url
                    )

            else:
                self.journal.add(
                    "error",
                    str(page)[:300],
                    question=question,
                    source=url
                )

        else:
            self.journal.add(
                "thought",
                "Подходящий адрес страницы не найден",
                question=question
            )

        self.curiosity.from_unknown_words(
            goal.text
        )

        self.goals.complete(
            goal,
            result=(
                "Материал сохранён для проверки"
                if saved
                else "Поиск завершён без сохранения материала"
            )
        )

        self.journal.add(
            "thought",
            "Цель обработана",
            goal=goal.text,
            question=question,
            saved=saved
        )

        self.learner.maybe_train(
            self.step
        )

    @staticmethod
    def _extract_topic(
        goal_text: str
    ) -> Optional[str]:
        if not isinstance(goal_text, str):
            return None

        text = " ".join(
            goal_text.split()
        )

        if not text:
            return None

        prefixes = (
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
            "полезный совет по "
        )

        normalized = text.casefold()

        for prefix in prefixes:
            if normalized.startswith(prefix):
                result = text[len(prefix):].strip()
                return result or None

        return text

    @staticmethod
    def _first_url(
        search_text: str
    ) -> Optional[str]:
        if not isinstance(search_text, str):
            return None

        for line in search_text.splitlines():
            line = line.strip()

            if not line.startswith(
                ("http://", "https://")
            ):
                continue

            url = decode_ddg_url(
                line
            )

            if is_public_url(
                url,
                resolve_dns=False
            ):
                return url

        return None

    def run(
        self,
        max_steps: Optional[int] = None
    ) -> None:
        if max_steps is None:
            max_steps = Config.LIFE_MAX_STEPS_PER_SESSION

        max_steps = max(
            0,
            min(
                int(max_steps),
                int(Config.LIFE_MAX_STEPS_PER_SESSION)
            )
        )

        step_pause = max(
            0.0,
            float(Config.LIFE_STEP_PAUSE_SEC)
        )

        sleep_interval = max(
            0,
            int(Config.LIFE_SLEEP_EVERY_STEPS)
        )

        sleep_duration = max(
            0.0,
            float(Config.LIFE_SLEEP_SEC)
        )

        print("=" * 60)
        print("KoKs: жизнь началась")
        print(f"Шагов за сессию: {max_steps}")
        print(f"Пауза между шагами: {step_pause} с")
        print("=" * 60)

        self.journal.add(
            "thought",
            "Проснулся"
        )

        try:
            for index in range(max_steps):
                try:
                    self._do_step()

                except KeyboardInterrupt:
                    raise

                except Exception as error:
                    self.journal.add(
                        "error",
                        f"Ошибка шага: {error}",
                        step=self.step
                    )

                if index + 1 >= max_steps:
                    break

                if (
                    sleep_interval > 0
                    and self.step > 0
                    and self.step % sleep_interval == 0
                ):
                    self.journal.add(
                        "thought",
                        f"Сон на {sleep_duration} секунд"
                    )

                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

                elif step_pause > 0:
                    time.sleep(step_pause)

        except KeyboardInterrupt:
            self.journal.add(
                "thought",
                "Остановлен пользователем"
            )

            print(
                "\nЖизнь остановлена пользователем."
            )

        finally:
            self.journal.add(
                "thought",
                "Уснул"
            )

            self.goals.save()

            stats = self.journal.stats()

            print("=" * 60)
            print("Статистика журнала:")

            for kind, count in stats.items():
                print(
                    f"{kind}: {count}"
                )

            print("=" * 60)
