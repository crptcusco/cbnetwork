class CustomText:
    @staticmethod
    def print_duplex_line():
        """
        Print a line of equal signs for separation or emphasis.
        """
        print(50 * '=')

    @staticmethod
    def print_simple_line():
        """
        Print a simple line of hyphens for separation or emphasis.
        """
        print("-" * 50)

    @staticmethod
    def print_message(message, show):
        """
        Print a message if the 'show' parameter is True.

        Args:
            message (str): The message to print.
            show (bool): Flag indicating whether to print the message.
        """
        if show:
            print(message)

    @staticmethod
    def print_stars():
        """
        Print a line of asterisks for separation or emphasis.
        """
        print(50 * '*')

    @staticmethod
    def print_dollars():
        """
        Print a line of dollar signs for separation or emphasis.
        """
        print(50 * '$')

    @staticmethod
    def make_principal_title(title):
        """
        Print a principal title surrounded by dollar signs.

        Args:
            title (str): The title to print.
        """
        print(50 * '$')
        print(title.upper())

    @staticmethod
    def make_title(title):
        """
        Print a title surrounded by asterisks.

        Args:
            title (str): The title to print.
        """
        print(50 * '*')
        print(title.upper())

    @staticmethod
    def make_sub_title(sub_title):
        """
        Print a subtitle surrounded by equal signs.

        Args:
            sub_title (str): The subtitle to print.
        """
        print(50 * '=')
        print(sub_title.upper())

    @staticmethod
    def make_sub_sub_title(sub_sub_title):
        """
        Print a sub-subtitle surrounded by hyphens.

        Args:
            sub_sub_title (str): The sub-subtitle to print.
        """
        print(50 * '-')
        print(sub_sub_title)

    @staticmethod
    def send_warning(message):
        """
        Print a warning message.

        Args:
            message (str): The warning message to print.
        """
        print('WARNING:', message)
