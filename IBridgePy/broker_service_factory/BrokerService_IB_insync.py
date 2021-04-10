# coding=utf-8
from IBridgePy.constants import BrokerServiceName, BrokerName
from broker_client_factory.BrokerClientDefs import ReqAccountUpdates
from broker_service_factory.BrokerService_web import WebApi


class IBinsync(WebApi):
    def get_timestamp(self, security, tickType):
        raise NotImplementedError

    def get_contract_details(self, security, field):
        raise NotImplementedError

    @property
    def name(self):
        return BrokerServiceName.IBinsync

    @property
    def brokerName(self):
        return BrokerName.IB

    # noinspection DuplicatedCode
    def _get_account_info_one_tag(self, accountCode, tag, meta='value'):
        self._log.notset(__name__ + '::_get_account_info_one_tag: accountCode=%s tag=%s' % (accountCode, tag))
        submitted = self._submit_request_after_checking_cache(ReqAccountUpdates(True, accountCode))
        # noinspection DuplicatedCode
        ans = self._singleTrader.get_account_info(self.brokerName, accountCode, tag)
        if ans is None:
            self._log.error(__name__ + '::_get_account_info_one_tag: EXIT, no value based on accountCode=%s tag=%s' % (accountCode, tag))
            print('active accountCode is %s' % (self._singleTrader.get_all_active_accountCodes(self.name),))
            exit()
        if submitted:
            self.submit_requests(ReqAccountUpdates(False, accountCode))
        return ans
