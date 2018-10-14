from core.communication_manager import CommunicationManager

configmgr = None
commmgr = CommunicationManager(configmgr)
commmgr.inform("new_session", {'key': 
                                'QmSCWJf7pX5rDdQbgsRmU2wARwinEJENpsGWPwV1eAcVvA', 
                                'content': 
                                {'kwargs': 'booga booga'}
                                }
                )
commmgr.inform("done_splitting", {'key': 'nonsense',
                                    'content': 'more nonsense'})
commmgr.inform("new_weights", {'key': 
                                'QmSCWJf7pX5rDdQbgsRmU2wARwinEJENpsGWPwV1eAcVvA', 
                                'content': 
                                {'weights': '[1,2],[3,4]'}
                                }
                )