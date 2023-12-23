class CustomText:
    @staticmethod
    def print_duplex_line():
        print("=================================================")

    @staticmethod
    def print_simple_line():
        print("-------------------------------------------------")

    @staticmethod
    def print_message(message, show):
        if show:
            print(message)

