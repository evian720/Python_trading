# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 23:50:16 2017

@author: IBridgePy@gmail.com
"""
from sys import exit

from IBridgePy.constants import BrokerClientName


class Converter(object):
    """
    IB uses integer as orderId and it must increase.
    Other brokers use string as orderId.
    And IBridgePy has switched fro int_orderId to str_orderId.
    For IB, str_orderId = 'ib' + int_orderId
    For other brokers, use broker's original str_orderId
    """

    def __init__(self, brokerClientName, createrOfIBValue=None):
        self._brokerClientName = brokerClientName
        self.fromBrokerToIBDict = {}
        self.fromIBToBrokerDict = {}
        self.createrOfIBValue = createrOfIBValue

    def fromBrokerToIB(self, brokerValue):
        """
        Converter a str_orderId to int_orderId
        :param brokerValue: string
        :return: int
        """
        if brokerValue in self.fromBrokerToIBDict:
            return self.fromBrokerToIBDict[brokerValue]
        ibValue = None
        if self._brokerClientName in [BrokerClientName.IB, BrokerClientName.LOCAL, BrokerClientName.IBinsync]:
            ibValue = int(brokerValue)
        elif self._brokerClientName in [BrokerClientName.TD, BrokerClientName.ROBINHOOD]:
            ibValue = self.createrOfIBValue.useOne()
        else:
            print(__name__ + '::Converter::fromBrokerToIB: EXIT, cannot handle brokerServiceName=%s' % (self._brokerClientName,))
            exit()
        self.setRelationship(ibValue, brokerValue)
        return ibValue

    def fromIBtoBroker(self, ibValue):
        """
        Converter a int_orderId to str_orderId
        :param ibValue: int
        :return: string
        """
        # For non-IB orders, they should have been registered in brokerClient_xx using setRelationship
        if ibValue in self.fromIBToBrokerDict:
            return self.fromIBToBrokerDict[ibValue]

        if self._brokerClientName in [BrokerClientName.IB, BrokerClientName.LOCAL, BrokerClientName.IBinsync]:
            brokerValue = 'ib' + str(ibValue)
            self.setRelationship(ibValue, brokerValue)
            return brokerValue
        else:
            print(
                    __name__ + '::Converter::fromBrokerToIB: EXIT, For non-IB orders, they should have been registered in brokerClient_xx using setRelationship')
            exit()

    def setRelationship(self, ibValue, brokerValue):
        # print(__name__ + '::Converter::setRelationship: ibValue=%s brokerValue=%s' % (ibValue, brokerValue))
        self.fromBrokerToIBDict[brokerValue] = ibValue
        self.fromIBToBrokerDict[ibValue] = brokerValue

    def verifyRelationship(self, ibValue, brokerValue):
        ans = (ibValue in self.fromIBToBrokerDict) and (brokerValue in self.fromBrokerToIBDict) and (
                self.fromIBToBrokerDict[ibValue] == brokerValue)
        if not ans:
            print(__name__ + '::Converter::verifyRelationship: EXIT, ibValue=%s brokerValue=%s' % (
                ibValue, brokerValue))
            print(self.fromIBToBrokerDict)
            print(self.fromBrokerToIBDict)
            exit()
