import pytest
import random
import string
import re
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.sherlock import sherlock
from sherlock_project.notify import QueryNotify
from sherlock_project.result import QueryStatus
#from sherlock_interactives import Interactives


async def simple_query(sites_info: dict, site: str, username: str, playwright_engine: PlaywrightEngine) -> QueryStatus:
    query_notify = QueryNotify()
    site_data: dict = {}
    site_data[site] = sites_info[site]
    results_total = await sherlock(
        username=username,
        site_data=site_data,
        query_notify=query_notify,
        engine=playwright_engine
    )
    return results_total[site]['status'].status    

@pytest.mark.online
class TestLiveTargets:
    """Actively test probes against live and trusted targets"""
    # Known positives should only use sites trusted to be reliable and unchanging
    @pytest.mark.parametrize('site,username',[
        ('GitLab', 'ppfeister'),
        ('AllMyLinks', 'blue'),
    ])
    @pytest.mark.asyncio()
    async def test_known_positives_via_message(self, sites_info, site, username, playwright_engine):
        assert await simple_query(sites_info=sites_info, site=site, username=username, playwright_engine=playwright_engine) is QueryStatus.CLAIMED


    # Known positives should only use sites trusted to be reliable and unchanging
    @pytest.mark.parametrize('site,username',[
        ('GitHub', 'ppfeister'),
        ('GitHub', 'sherlock-project'),
        ('Docker Hub', 'ppfeister'),
        ('Docker Hub', 'sherlock'),
    ])
    @pytest.mark.asyncio()
    async def test_known_positives_via_status_code(self, sites_info, site, username, playwright_engine):
        assert await simple_query(sites_info=sites_info, site=site, username=username, playwright_engine=playwright_engine) is QueryStatus.CLAIMED


    # Known positives should only use sites trusted to be reliable and unchanging
    @pytest.mark.parametrize('site,username',[
        ('Keybase', 'blue'),
        ('devRant', 'blue'),
    ])
    @pytest.mark.asyncio()
    async def test_known_positives_via_response_url(self, sites_info, site, username, playwright_engine):
        assert await simple_query(sites_info=sites_info, site=site, username=username, playwright_engine=playwright_engine) is QueryStatus.CLAIMED


    # Randomly generate usernames of high length and test for positive availability
    # Randomly generated usernames should be simple alnum for simplicity and high
    # compatibility. Several attempts may be made ~just in case~ a real username is
    # generated.
    @pytest.mark.parametrize('site,random_len',[
        ('GitLab', 255),
        ('Codecademy', 30)
    ])
    @pytest.mark.asyncio()
    async def test_likely_negatives_via_message(self, sites_info, site, random_len, playwright_engine):
        num_attempts: int = 3
        attempted_usernames: list[str] = []
        status: QueryStatus = QueryStatus.CLAIMED
        for i in range(num_attempts):
            acceptable_types = string.ascii_letters + string.digits
            random_handle = ''.join(random.choice(acceptable_types) for _ in range (random_len))
            attempted_usernames.append(random_handle)
            status = await simple_query(sites_info=sites_info, site=site, username=random_handle, playwright_engine=playwright_engine)
            if status is QueryStatus.AVAILABLE:
                break
        assert status is QueryStatus.AVAILABLE, f"Could not validate available username after {num_attempts} attempts with randomly generated usernames {attempted_usernames}."


    # Randomly generate usernames of high length and test for positive availability
    # Randomly generated usernames should be simple alnum for simplicity and high
    # compatibility. Several attempts may be made ~just in case~ a real username is
    # generated.
    @pytest.mark.parametrize('site,random_len',[
        ('GitHub', 39),
        ('Docker Hub', 30)
    ])
    @pytest.mark.asyncio()
    async def test_likely_negatives_via_status_code(self, sites_info, site, random_len, playwright_engine):
        num_attempts: int = 3
        attempted_usernames: list[str] = []
        status: QueryStatus = QueryStatus.CLAIMED
        for i in range(num_attempts):
            acceptable_types = string.ascii_letters + string.digits
            random_handle = ''.join(random.choice(acceptable_types) for _ in range (random_len))
            attempted_usernames.append(random_handle)
            status = await simple_query(sites_info=sites_info, site=site, username=random_handle, playwright_engine=playwright_engine)
            if status is QueryStatus.AVAILABLE:
                break
        assert status is QueryStatus.AVAILABLE, f"Could not validate available username after {num_attempts} attempts with randomly generated usernames {attempted_usernames}."

@pytest.mark.asyncio()
async def test_username_illegal_regex(sites_info, playwright_engine):
    site: str = 'Bitwarden Forum'
    invalid_handle: str = '*#$Y&*JRE'
    pattern = re.compile(sites_info[site]['regexCheck'])
    # Ensure that the username actually fails regex before testing sherlock
    assert pattern.match(invalid_handle) is None
    assert await simple_query(sites_info=sites_info, site=site, username=invalid_handle, playwright_engine=playwright_engine) is QueryStatus.ILLEGAL

