class CustomText:
    @staticmethod
    def print_duplex_line():
        print("=================================================")

    @staticmethod
    def print_simple_line():
        print("-"*50)

    @staticmethod
    def print_message(message, show):
        if show:
            print(message)

    @staticmethod
    def print_stars():
        print("*************************************************")

    @staticmethod
    def print_dollars():
        print(50 * '$')

    @staticmethod
    def make_principal_title(title):
        print(50 * '$')
        print(title.upper())

    @staticmethod
    def make_title(title):
        print(50 * '*')
        print(title.upper())

    @staticmethod
    def make_sub_title(sub_title):
        print(50 * '=')
        print(sub_title.upper())

    @staticmethod
    def make_sub_sub_title(sub_sub_title):
        print(50 * '-')
        print(sub_sub_title)


