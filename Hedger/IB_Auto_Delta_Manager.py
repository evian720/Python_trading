# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:20:20 2021

@author: Evian Zhou
"""


def initialize(context):
	context.security = symbol('GME')
	context.max_size = 600

	context.hedging_price = 165
	context.unwind_hedging_price_factor = 0.995

	context.hedging_size = 200

	context.run_once = False
	context.hedge_executed_once = False
	context.orderID_list = []


def handle_data(context, data):
	if context.run_once == False:

		portfolio_value = show_account_info("portfolio_value")
		print("\nValue of the portfolio:", portfolio_value, "\n")

		positions_value = show_account_info("positions_value")
		print("\nValue of the positions:", positions_value, "\n")

		account_cash = show_account_info("cash")
		print("\nCash in the portfolio:", account_cash,"\n")

		cancel_all_orders()

		context.run_once = True


	print (get_datetime().strftime("%Y-%m-%d %H:%M:%S %Z"))

	position = context.portfolio.positions[context.security].amount
	last_price = show_real_time_price(context.security, "last_price")

	print("Holding", position, "shs of", context.security.symbol , ". Last Price:" , last_price)
	print("Monitoring Price:", context.hedging_price, " Size:" , context.hedging_size)


	if position <= context.max_size:

		# Buy to target size
		if last_price != "-1" and last_price > context.hedging_price and position < context.hedging_size:
			print("Strike Price breached. Doing hedging now.")
			context.hedge_executed_once = True
			unwind_price = round(context.hedging_price * context.unwind_hedging_price_factor,2)
			orderId = place_order_with_stoploss (context.security, context.hedging_size, unwind_price, style=MarketOrder())

			for i in orderId:
				print("Order ID:", i, "Status: ", get_order_status(i))
				# http://www.ibridgepy.com/ibridgepy-documentation/#order_status_monitor
				order_status_monitor(i, target_status=['Filled', 'Submitted', 'PreSubmitted'])
				context.orderID_list.append(i)

	if(context.hedge_executed_once == True):
		for i in context.orderID_list:
			print("Order ID:", i, "Status: ", get_order_status(i))