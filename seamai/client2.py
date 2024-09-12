import logging, sys, os, warnings

# Ignore user warnings
warnings.simplefilter(action="ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the system path
sys.path.append('../')

import config as cfg
from events.Client import Client


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Function to initialize the client
def initialize(first=True):
    """
    Initializes a Client object.

    Parameters:
    first (bool): Indicates whether this is the first initialization.

    Returns:
    Client: An initialized Client object.
    """
    logger.info('Preparing Client.')

    # Creating environment
    if cfg.SERVER_ADDR == '127.0.0.1':
        # If the server address is localhost, create a Client with a unique source address
        client = Client(
            0,
            cfg.SERVER_ADDR,
            cfg.SERVER_PORT,
            cfg.limit_bw,
            source_addr='127.0.0.' + str(int((os.path.basename(__file__).split('.')[0]).split('client')[-1]) + 1)
        )
    else:
        # Otherwise, create a Client with the provided server address
        client = Client(
            0,
            cfg.SERVER_ADDR,
            cfg.SERVER_PORT,
            cfg.limit_bw,
        source_addr = cfg.SOURCE_ADDR[int((os.path.basename(__file__).split('.')[0]).split('client')[-1]) - 1]
        )

    # Initialize the client
    client.initialize(first=first)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Return the initialized client
    return client


# Main function to run the client
def main():
    """
    Main function to run the Client in a loop for multiple rounds.
    """
    client = initialize(True)
    while client.r + 1:
        try:
            logger.info('\n#####################################################################\n')
            logger.info('ROUND: {} START'.format(client.r))

            # Set client for the current round
            client.set()

            logger.info('ROUND: {} END'.format(client.r))

        except BaseException as e:
            logger.info(e)

            # Close the client socket and reinitialize the client
            client.sock.close()
            client.reinitialize(True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
