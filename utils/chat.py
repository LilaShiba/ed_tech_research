
class ChatBot:
    """ Chat with resources """

    def __init__(self, name, quest=None) -> None:
        """ Chat with resources """

        self.name = name
        self.current_question = quest

    def question(self, quest):
        """ Chat with resources """

        self.current_question = quest

    def enter_chat(self, quest=None):
        """ Start Chat with resources """
        if not quest:
            exit_flag = False
            while not exit_flag:
                quest = input(f"Please ask a Questions about {self.name}")
