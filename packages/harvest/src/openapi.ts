
            export default {
  "openapi": "3.0.1",
  "info": {
    "version": "1.0.0",
    "title": "Harvestapp API",
    "license": {
      "name": "MIT"
    }
  },
  "externalDocs": {
    "description": "Learn more about the Harvest Web API",
    "url": "https://help.getharvest.com/api-v2/"
  },
  "servers": [
    {
      "url": "https://api.harvestapp.com/v2"
    }
  ],
  "components": {
    "securitySchemes": {
      "BearerAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "Authorization"
      },
      "AccountAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "Harvest-Account-Id"
      }
    },
    "schemas": {
      "Contact": {
        "type": "object",
        "externalDocs": {
          "description": "contact",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/contacts/#the-contact-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the contact.",
            "nullable": true,
            "format": "int32"
          },
          "client": {
            "type": "object",
            "description": "An object containing the contact’s client id and name.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "title": {
            "type": "string",
            "description": "The title of the contact.",
            "nullable": true
          },
          "first_name": {
            "type": "string",
            "description": "The first name of the contact.",
            "nullable": true
          },
          "last_name": {
            "type": "string",
            "description": "The last name of the contact.",
            "nullable": true
          },
          "email": {
            "type": "string",
            "description": "The contact’s email address.",
            "nullable": true,
            "format": "email"
          },
          "phone_office": {
            "type": "string",
            "description": "The contact’s office phone number.",
            "nullable": true
          },
          "phone_mobile": {
            "type": "string",
            "description": "The contact’s mobile phone number.",
            "nullable": true
          },
          "fax": {
            "type": "string",
            "description": "The contact’s fax number.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the contact was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the contact was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Client": {
        "type": "object",
        "externalDocs": {
          "description": "client",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/clients/#the-client-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the client.",
            "nullable": true,
            "format": "int32"
          },
          "name": {
            "type": "string",
            "description": "A textual description of the client.",
            "nullable": true
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the client is active or archived.",
            "nullable": true
          },
          "address": {
            "type": "string",
            "description": "The physical address for the client.",
            "nullable": true
          },
          "statement_key": {
            "type": "string",
            "description": "Used to build a URL to your client’s invoice dashboard:https://{ACCOUNT_SUBDOMAIN}.harvestapp.com/client/statements/{STATEMENT_KEY}",
            "nullable": true
          },
          "currency": {
            "type": "string",
            "description": "The currency code associated with this client.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the client was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the client was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Company": {
        "type": "object",
        "externalDocs": {
          "description": "company",
          "url": "https://help.getharvest.com/api-v2/company-api/company/company/#the-company-object"
        },
        "properties": {
          "base_uri": {
            "type": "string",
            "description": "The Harvest URL for the company.",
            "nullable": true
          },
          "full_domain": {
            "type": "string",
            "description": "The Harvest domain for the company.",
            "nullable": true
          },
          "name": {
            "type": "string",
            "description": "The name of the company.",
            "nullable": true
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the company is active or archived.",
            "nullable": true
          },
          "week_start_day": {
            "type": "string",
            "description": "The weekday used as the start of the week. Returns one of: Saturday, Sunday, or Monday.",
            "nullable": true
          },
          "wants_timestamp_timers": {
            "type": "boolean",
            "description": "Whether time is tracked via duration or start and end times.",
            "nullable": true
          },
          "time_format": {
            "type": "string",
            "description": "The format used to display time in Harvest. Returns either decimal or hours_minutes.",
            "nullable": true
          },
          "date_format": {
            "type": "string",
            "description": "The format used to display date in Harvest. Returns one of: %m/%d/%Y, %d/%m/%Y, %Y-%m-%d, %d.%m.%Y,.%Y.%m.%d or %Y/%m/%d.",
            "nullable": true
          },
          "plan_type": {
            "type": "string",
            "description": "The type of plan the company is on. Examples: trial, free, or simple-v4",
            "nullable": true
          },
          "clock": {
            "type": "string",
            "description": "Used to represent whether the company is using a 12-hour or 24-hour clock. Returns either 12h or 24h.",
            "nullable": true
          },
          "currency_code_display": {
            "type": "string",
            "description": "How to display the currency code when formatting currency. Returns one of: iso_code_none, iso_code_before, or iso_code_after.",
            "nullable": true
          },
          "currency_symbol_display": {
            "type": "string",
            "description": "How to display the currency symbol when formatting currency. Returns one of: symbol_none, symbol_before, or symbol_after.",
            "nullable": true
          },
          "decimal_symbol": {
            "type": "string",
            "description": "Symbol used when formatting decimals.",
            "nullable": true
          },
          "thousands_separator": {
            "type": "string",
            "description": "Separator used when formatting numbers.",
            "nullable": true
          },
          "color_scheme": {
            "type": "string",
            "description": "The color scheme being used in the Harvest web client.",
            "nullable": true
          },
          "weekly_capacity": {
            "type": "integer",
            "description": "The weekly capacity in seconds.",
            "nullable": true,
            "format": "int32"
          },
          "expense_feature": {
            "type": "boolean",
            "description": "Whether the expense module is enabled.",
            "nullable": true
          },
          "invoice_feature": {
            "type": "boolean",
            "description": "Whether the invoice module is enabled.",
            "nullable": true
          },
          "estimate_feature": {
            "type": "boolean",
            "description": "Whether the estimate module is enabled.",
            "nullable": true
          },
          "approval_feature": {
            "type": "boolean",
            "description": "Whether the approval module is enabled.",
            "nullable": true
          }
        }
      },
      "InvoiceMessage": {
        "type": "object",
        "externalDocs": {
          "description": "invoice-message",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-messages/#the-invoice-message-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the message.",
            "nullable": true,
            "format": "int32"
          },
          "sent_by": {
            "type": "string",
            "description": "Name of the user that created the message.",
            "nullable": true
          },
          "sent_by_email": {
            "type": "string",
            "description": "Email of the user that created the message.",
            "nullable": true
          },
          "sent_from": {
            "type": "string",
            "description": "Name of the user that the message was sent from.",
            "nullable": true
          },
          "sent_from_email": {
            "type": "string",
            "description": "Email of the user that message was sent from.",
            "nullable": true
          },
          "recipients": {
            "type": "array",
            "description": "Array of invoice message recipients.",
            "nullable": true,
            "items": {
              "$ref": "#/components/schemas/InvoiceMessageRecipient"
            }
          },
          "subject": {
            "type": "string",
            "description": "The message subject.",
            "nullable": true
          },
          "body": {
            "type": "string",
            "description": "The message body.",
            "nullable": true
          },
          "include_link_to_client_invoice": {
            "type": "boolean",
            "description": "DEPRECATED This will be true when payment_options are assigned to the invoice and false when there are no payment_options.",
            "nullable": true,
            "deprecated": true
          },
          "attach_pdf": {
            "type": "boolean",
            "description": "Whether to attach the invoice PDF to the message email.",
            "nullable": true
          },
          "send_me_a_copy": {
            "type": "boolean",
            "description": "Whether to email a copy of the message to the current user.",
            "nullable": true
          },
          "thank_you": {
            "type": "boolean",
            "description": "Whether this is a thank you message.",
            "nullable": true
          },
          "event_type": {
            "type": "string",
            "description": "The type of invoice event that occurred with the message: send, close, draft, re-open, or view.",
            "nullable": true
          },
          "reminder": {
            "type": "boolean",
            "description": "Whether this is a reminder message.",
            "nullable": true
          },
          "send_reminder_on": {
            "type": "string",
            "description": "The date the reminder email will be sent.",
            "nullable": true,
            "format": "date"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the message was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the message was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "InvoiceMessageRecipient": {
        "type": "object",
        "externalDocs": {
          "description": "invoice-message-recipient",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-messages/#the-invoice-message-recipient-object"
        },
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the message recipient.",
            "nullable": true
          },
          "email": {
            "type": "string",
            "description": "Email of the message recipient.",
            "nullable": true,
            "format": "email"
          }
        }
      },
      "InvoicePayment": {
        "type": "object",
        "externalDocs": {
          "description": "invoice-payment",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-payments/#the-invoice-payment-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the payment.",
            "nullable": true,
            "format": "int32"
          },
          "amount": {
            "type": "number",
            "description": "The amount of the payment.",
            "nullable": true,
            "format": "float"
          },
          "paid_at": {
            "type": "string",
            "description": "Date and time the payment was made.",
            "nullable": true,
            "format": "date-time"
          },
          "paid_date": {
            "type": "string",
            "description": "Date the payment was made.",
            "nullable": true,
            "format": "date"
          },
          "recorded_by": {
            "type": "string",
            "description": "The name of the person who recorded the payment.",
            "nullable": true
          },
          "recorded_by_email": {
            "type": "string",
            "description": "The email of the person who recorded the payment.",
            "nullable": true
          },
          "notes": {
            "type": "string",
            "description": "Any notes associated with the payment.",
            "nullable": true
          },
          "transaction_id": {
            "type": "string",
            "description": "Either the card authorization or PayPal transaction ID.",
            "nullable": true
          },
          "payment_gateway": {
            "type": "object",
            "description": "The payment gateway id and name used to process the payment.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the payment was recorded.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the payment was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Invoice": {
        "type": "object",
        "externalDocs": {
          "description": "invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoices/#the-invoice-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the invoice.",
            "nullable": true,
            "format": "int32"
          },
          "client": {
            "type": "object",
            "description": "An object containing invoice’s client id and name.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "line_items": {
            "type": "array",
            "description": "Array of invoice line items.",
            "nullable": true,
            "items": {
              "$ref": "#/components/schemas/InvoiceLineItem"
            }
          },
          "estimate": {
            "type": "object",
            "description": "An object containing the associated estimate’s id.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              }
            }
          },
          "retainer": {
            "type": "object",
            "description": "An object containing the associated retainer’s id.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              }
            }
          },
          "creator": {
            "type": "object",
            "description": "An object containing the id and name of the person that created the invoice.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "client_key": {
            "type": "string",
            "description": "Used to build a URL to the public web invoice for your client by adding /client/invoices/{CLIENT_KEY} to your account URL https://{SUBDOMAIN}.harvestapp.com/ Note: you can also add .pdf to the end of this URL to access a PDF version of the invoice.",
            "nullable": true
          },
          "number": {
            "type": "string",
            "description": "If no value is set, the number will be automatically generated.",
            "nullable": true
          },
          "purchase_order": {
            "type": "string",
            "description": "The purchase order number.",
            "nullable": true
          },
          "amount": {
            "type": "number",
            "description": "The total amount for the invoice, including any discounts and taxes.",
            "nullable": true,
            "format": "float"
          },
          "due_amount": {
            "type": "number",
            "description": "The total amount due at this time for this invoice.",
            "nullable": true,
            "format": "float"
          },
          "tax": {
            "type": "number",
            "description": "This percentage is applied to the subtotal, including line items and discounts.",
            "nullable": true,
            "format": "float"
          },
          "tax_amount": {
            "type": "number",
            "description": "The first amount of tax included, calculated from tax. If no tax is defined, this value will be null.",
            "nullable": true,
            "format": "float"
          },
          "tax2": {
            "type": "number",
            "description": "This percentage is applied to the subtotal, including line items and discounts.",
            "nullable": true,
            "format": "float"
          },
          "tax2_amount": {
            "type": "number",
            "description": "The amount calculated from tax2.",
            "nullable": true,
            "format": "float"
          },
          "discount": {
            "type": "number",
            "description": "This percentage is subtracted from the subtotal.",
            "nullable": true,
            "format": "float"
          },
          "discount_amount": {
            "type": "number",
            "description": "The amount calculated from discount.",
            "nullable": true,
            "format": "float"
          },
          "subject": {
            "type": "string",
            "description": "The invoice subject.",
            "nullable": true
          },
          "notes": {
            "type": "string",
            "description": "Any additional notes included on the invoice.",
            "nullable": true
          },
          "currency": {
            "type": "string",
            "description": "The currency code associated with this invoice.",
            "nullable": true
          },
          "state": {
            "type": "string",
            "description": "The current state of the invoice: draft, open, paid, or closed.",
            "nullable": true
          },
          "period_start": {
            "type": "string",
            "description": "Start of the period during which time entries were added to this invoice.",
            "nullable": true,
            "format": "date"
          },
          "period_end": {
            "type": "string",
            "description": "End of the period during which time entries were added to this invoice.",
            "nullable": true,
            "format": "date"
          },
          "issue_date": {
            "type": "string",
            "description": "Date the invoice was issued.",
            "nullable": true,
            "format": "date"
          },
          "due_date": {
            "type": "string",
            "description": "Date the invoice is due.",
            "nullable": true,
            "format": "date"
          },
          "payment_term": {
            "type": "string",
            "description": "The timeframe in which the invoice should be paid. Options: upon receipt, net 15, net 30, net 45, net 60, or custom.",
            "nullable": true
          },
          "payment_options": {
            "type": "array",
            "description": "The list of payment options enabled for the invoice. Options: [ach, credit_card, paypal]",
            "nullable": true,
            "items": {
              "type": "string",
              "enum": [
                "ach",
                "credit_card",
                "paypal"
              ]
            }
          },
          "sent_at": {
            "type": "string",
            "description": "Date and time the invoice was sent.",
            "nullable": true,
            "format": "date-time"
          },
          "paid_at": {
            "type": "string",
            "description": "Date and time the invoice was paid.",
            "nullable": true,
            "format": "date-time"
          },
          "paid_date": {
            "type": "string",
            "description": "Date the invoice was paid.",
            "nullable": true,
            "format": "date"
          },
          "closed_at": {
            "type": "string",
            "description": "Date and time the invoice was closed.",
            "nullable": true,
            "format": "date-time"
          },
          "recurring_invoice_id": {
            "type": "integer",
            "description": "Unique ID of the associated recurring invoice.",
            "nullable": true,
            "format": "int32"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the invoice was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the invoice was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "InvoiceLineItem": {
        "type": "object",
        "externalDocs": {
          "description": "invoice-line-item",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoices/#the-invoice-line-item-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the line item.",
            "nullable": true,
            "format": "int32"
          },
          "project": {
            "type": "object",
            "description": "An object containing the associated project’s id, name, and code.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "code": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "kind": {
            "type": "string",
            "description": "The name of an invoice item category.",
            "nullable": true
          },
          "description": {
            "type": "string",
            "description": "Text description of the line item.",
            "nullable": true
          },
          "quantity": {
            "type": "number",
            "description": "The unit quantity of the item.",
            "nullable": true,
            "format": "float"
          },
          "unit_price": {
            "type": "number",
            "description": "The individual price per unit.",
            "nullable": true,
            "format": "float"
          },
          "amount": {
            "type": "number",
            "description": "The line item subtotal (quantity * unit_price).",
            "nullable": true,
            "format": "float"
          },
          "taxed": {
            "type": "boolean",
            "description": "Whether the invoice’s tax percentage applies to this line item.",
            "nullable": true
          },
          "taxed2": {
            "type": "boolean",
            "description": "Whether the invoice’s tax2 percentage applies to this line item.",
            "nullable": true
          }
        }
      },
      "InvoiceItemCategory": {
        "type": "object",
        "externalDocs": {
          "description": "invoice-item-category",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-item-categories/#the-invoice-item-category-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the invoice item category.",
            "nullable": true,
            "format": "int32"
          },
          "name": {
            "type": "string",
            "description": "The name of the invoice item category.",
            "nullable": true
          },
          "use_as_service": {
            "type": "boolean",
            "description": "Whether this invoice item category is used for billable hours when generating an invoice.",
            "nullable": true
          },
          "use_as_expense": {
            "type": "boolean",
            "description": "Whether this invoice item category is used for expenses when generating an invoice.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the invoice item category was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the invoice item category was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "EstimateMessage": {
        "type": "object",
        "externalDocs": {
          "description": "estimate-message",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-messages/#the-estimate-message-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the message.",
            "nullable": true,
            "format": "int32"
          },
          "sent_by": {
            "type": "string",
            "description": "Name of the user that created the message.",
            "nullable": true
          },
          "sent_by_email": {
            "type": "string",
            "description": "Email of the user that created the message.",
            "nullable": true
          },
          "sent_from": {
            "type": "string",
            "description": "Name of the user that the message was sent from.",
            "nullable": true
          },
          "sent_from_email": {
            "type": "string",
            "description": "Email of the user that message was sent from.",
            "nullable": true
          },
          "recipients": {
            "type": "array",
            "description": "Array of estimate message recipients.",
            "nullable": true,
            "items": {
              "$ref": "#/components/schemas/EstimateMessageRecipient"
            }
          },
          "subject": {
            "type": "string",
            "description": "The message subject.",
            "nullable": true
          },
          "body": {
            "type": "string",
            "description": "The message body.",
            "nullable": true
          },
          "send_me_a_copy": {
            "type": "boolean",
            "description": "Whether to email a copy of the message to the current user.",
            "nullable": true
          },
          "event_type": {
            "type": "string",
            "description": "The type of estimate event that occurred with the message: send, accept, decline, re-open, view, or invoice.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the message was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the message was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "EstimateMessageRecipient": {
        "type": "object",
        "externalDocs": {
          "description": "estimate-message-recipient",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-messages/#the-estimate-message-recipient-object"
        },
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the message recipient.",
            "nullable": true
          },
          "email": {
            "type": "string",
            "description": "Email of the message recipient.",
            "nullable": true,
            "format": "email"
          }
        }
      },
      "Estimate": {
        "type": "object",
        "externalDocs": {
          "description": "estimate",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimates/#the-estimate-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the estimate.",
            "nullable": true,
            "format": "int32"
          },
          "client": {
            "type": "object",
            "description": "An object containing estimate’s client id and name.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "line_items": {
            "type": "array",
            "description": "Array of estimate line items.",
            "nullable": true,
            "items": {
              "$ref": "#/components/schemas/EstimateLineItem"
            }
          },
          "creator": {
            "type": "object",
            "description": "An object containing the id and name of the person that created the estimate.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "client_key": {
            "type": "string",
            "description": "Used to build a URL to the public web invoice for your client:https://{ACCOUNT_SUBDOMAIN}.harvestapp.com/client/estimates/abc123456",
            "nullable": true
          },
          "number": {
            "type": "string",
            "description": "If no value is set, the number will be automatically generated.",
            "nullable": true
          },
          "purchase_order": {
            "type": "string",
            "description": "The purchase order number.",
            "nullable": true
          },
          "amount": {
            "type": "number",
            "description": "The total amount for the estimate, including any discounts and taxes.",
            "nullable": true,
            "format": "float"
          },
          "tax": {
            "type": "number",
            "description": "This percentage is applied to the subtotal, including line items and discounts.",
            "nullable": true,
            "format": "float"
          },
          "tax_amount": {
            "type": "number",
            "description": "The first amount of tax included, calculated from tax. If no tax is defined, this value will be null.",
            "nullable": true,
            "format": "float"
          },
          "tax2": {
            "type": "number",
            "description": "This percentage is applied to the subtotal, including line items and discounts.",
            "nullable": true,
            "format": "float"
          },
          "tax2_amount": {
            "type": "number",
            "description": "The amount calculated from tax2.",
            "nullable": true,
            "format": "float"
          },
          "discount": {
            "type": "number",
            "description": "This percentage is subtracted from the subtotal.",
            "nullable": true,
            "format": "float"
          },
          "discount_amount": {
            "type": "number",
            "description": "The amount calculated from discount.",
            "nullable": true,
            "format": "float"
          },
          "subject": {
            "type": "string",
            "description": "The estimate subject.",
            "nullable": true
          },
          "notes": {
            "type": "string",
            "description": "Any additional notes included on the estimate.",
            "nullable": true
          },
          "currency": {
            "type": "string",
            "description": "The currency code associated with this estimate.",
            "nullable": true
          },
          "state": {
            "type": "string",
            "description": "The current state of the estimate: draft, sent, accepted, or declined.",
            "nullable": true
          },
          "issue_date": {
            "type": "string",
            "description": "Date the estimate was issued.",
            "nullable": true,
            "format": "date"
          },
          "sent_at": {
            "type": "string",
            "description": "Date and time the estimate was sent.",
            "nullable": true,
            "format": "date-time"
          },
          "accepted_at": {
            "type": "string",
            "description": "Date and time the estimate was accepted.",
            "nullable": true,
            "format": "date-time"
          },
          "declined_at": {
            "type": "string",
            "description": "Date and time the estimate was declined.",
            "nullable": true,
            "format": "date-time"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the estimate was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the estimate was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "EstimateLineItem": {
        "type": "object",
        "externalDocs": {
          "description": "estimate-line-item",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimates/#the-estimate-line-item-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the line item.",
            "nullable": true,
            "format": "int32"
          },
          "kind": {
            "type": "string",
            "description": "The name of an estimate item category.",
            "nullable": true
          },
          "description": {
            "type": "string",
            "description": "Text description of the line item.",
            "nullable": true
          },
          "quantity": {
            "type": "number",
            "description": "The unit quantity of the item.",
            "nullable": true,
            "format": "float"
          },
          "unit_price": {
            "type": "number",
            "description": "The individual price per unit.",
            "nullable": true,
            "format": "float"
          },
          "amount": {
            "type": "number",
            "description": "The line item subtotal (quantity * unit_price).",
            "nullable": true,
            "format": "float"
          },
          "taxed": {
            "type": "boolean",
            "description": "Whether the estimate’s tax percentage applies to this line item.",
            "nullable": true
          },
          "taxed2": {
            "type": "boolean",
            "description": "Whether the estimate’s tax2 percentage applies to this line item.",
            "nullable": true
          }
        }
      },
      "EstimateItemCategory": {
        "type": "object",
        "externalDocs": {
          "description": "estimate-item-category",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-item-categories/#the-estimate-item-category-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the estimate item category.",
            "nullable": true,
            "format": "int32"
          },
          "name": {
            "type": "string",
            "description": "The name of the estimate item category.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the estimate item category was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the estimate item category was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Expense": {
        "type": "object",
        "externalDocs": {
          "description": "expense",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expenses/#the-expense-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the expense.",
            "nullable": true,
            "format": "int32"
          },
          "client": {
            "type": "object",
            "description": "An object containing the expense’s client id, name, and currency.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "currency": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "project": {
            "type": "object",
            "description": "An object containing the expense’s project id, name, and code.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "code": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "expense_category": {
            "type": "object",
            "description": "An object containing the expense’s expense category id, name, unit_price, and unit_name.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "unit_price": {
                "type": "string",
                "nullable": true
              },
              "unit_name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "user": {
            "type": "object",
            "description": "An object containing the id and name of the user that recorded the expense.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "user_assignment": {
            "type": "object",
            "description": "A user assignment object of the user that recorded the expense.",
            "nullable": true,
            "$ref": "#/components/schemas/UserAssignment"
          },
          "receipt": {
            "type": "object",
            "description": "An object containing the expense’s receipt URL and file name.",
            "nullable": true,
            "properties": {
              "url": {
                "type": "string",
                "nullable": true
              },
              "file_name": {
                "type": "string",
                "nullable": true
              },
              "file_size": {
                "type": "integer",
                "format": "int32",
                "nullable": true
              },
              "content_type": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "invoice": {
            "type": "object",
            "description": "Once the expense has been invoiced, this field will include the associated invoice’s id and number.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "number": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "notes": {
            "type": "string",
            "description": "Textual notes used to describe the expense.",
            "nullable": true
          },
          "units": {
            "type": "integer",
            "description": "The quantity of units used to calculate the total_cost of the expense.",
            "nullable": true,
            "format": "int32"
          },
          "total_cost": {
            "type": "number",
            "description": "The total amount of the expense.",
            "nullable": true,
            "format": "float"
          },
          "billable": {
            "type": "boolean",
            "description": "Whether the expense is billable or not.",
            "nullable": true
          },
          "is_closed": {
            "type": "boolean",
            "description": "Whether the expense has been approved or not.",
            "nullable": true
          },
          "is_locked": {
            "type": "boolean",
            "description": "Whether the expense has been been invoiced, approved, or the project or person related to the expense is archived.",
            "nullable": true
          },
          "is_billed": {
            "type": "boolean",
            "description": "Whether or not the expense has been marked as invoiced.",
            "nullable": true
          },
          "locked_reason": {
            "type": "string",
            "description": "An explanation of why the expense has been locked.",
            "nullable": true
          },
          "spent_date": {
            "type": "string",
            "description": "Date the expense occurred.",
            "nullable": true,
            "format": "date"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the expense was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the expense was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "ExpenseCategory": {
        "type": "object",
        "externalDocs": {
          "description": "expense-category",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expense-categories/#the-expense-category-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the expense category.",
            "nullable": true,
            "format": "int32"
          },
          "name": {
            "type": "string",
            "description": "The name of the expense category.",
            "nullable": true
          },
          "unit_name": {
            "type": "string",
            "description": "The unit name of the expense category.",
            "nullable": true
          },
          "unit_price": {
            "type": "number",
            "description": "The unit price of the expense category.",
            "nullable": true,
            "format": "float"
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the expense category is active or archived.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the expense category was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the expense category was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Task": {
        "type": "object",
        "externalDocs": {
          "description": "task",
          "url": "https://help.getharvest.com/api-v2/tasks-api/tasks/tasks/#the-task-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the task.",
            "nullable": true,
            "format": "int32"
          },
          "name": {
            "type": "string",
            "description": "The name of the task.",
            "nullable": true
          },
          "billable_by_default": {
            "type": "boolean",
            "description": "Used in determining whether default tasks should be marked billable when creating a new project.",
            "nullable": true
          },
          "default_hourly_rate": {
            "type": "number",
            "description": "The hourly rate to use for this task when it is added to a project.",
            "nullable": true,
            "format": "float"
          },
          "is_default": {
            "type": "boolean",
            "description": "Whether this task should be automatically added to future projects.",
            "nullable": true
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether this task is active or archived.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the task was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the task was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "TimeEntry": {
        "type": "object",
        "externalDocs": {
          "description": "time-entry",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#the-time-entry-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the time entry.",
            "nullable": true
          },
          "spent_date": {
            "type": "string",
            "description": "Date of the time entry.",
            "nullable": true,
            "format": "date"
          },
          "user": {
            "type": "object",
            "description": "An object containing the id and name of the associated user.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "user_assignment": {
            "type": "object",
            "description": "A user assignment object of the associated user.",
            "nullable": true,
            "$ref": "#/components/schemas/UserAssignment"
          },
          "client": {
            "type": "object",
            "description": "An object containing the id and name of the associated client.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "project": {
            "type": "object",
            "description": "An object containing the id and name of the associated project.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "task": {
            "type": "object",
            "description": "An object containing the id and name of the associated task.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "task_assignment": {
            "type": "object",
            "description": "A task assignment object of the associated task.",
            "nullable": true,
            "$ref": "#/components/schemas/TaskAssignment"
          },
          "external_reference": {
            "type": "object",
            "description": "An object containing the id, group_id, account_id, permalink, service, and service_icon_url of the associated external reference.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "string",
                "nullable": true
              },
              "group_id": {
                "type": "string",
                "nullable": true
              },
              "account_id": {
                "type": "string",
                "nullable": true
              },
              "permalink": {
                "type": "string",
                "nullable": true
              },
              "service": {
                "type": "string",
                "nullable": true
              },
              "service_icon_url": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "invoice": {
            "type": "object",
            "description": "Once the time entry has been invoiced, this field will include the associated invoice’s id and number.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "number": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "hours": {
            "type": "number",
            "description": "Number of (decimal time) hours tracked in this time entry.",
            "nullable": true,
            "format": "float"
          },
          "hours_without_timer": {
            "type": "number",
            "description": "Number of (decimal time) hours already tracked in this time entry, before the timer was last started.",
            "nullable": true,
            "format": "float"
          },
          "rounded_hours": {
            "type": "number",
            "description": "Number of (decimal time) hours tracked in this time entry used in summary reports and invoices. This value is rounded according to the Time Rounding setting in your Preferences.",
            "nullable": true,
            "format": "float"
          },
          "notes": {
            "type": "string",
            "description": "Notes attached to the time entry.",
            "nullable": true
          },
          "is_locked": {
            "type": "boolean",
            "description": "Whether or not the time entry has been locked.",
            "nullable": true
          },
          "locked_reason": {
            "type": "string",
            "description": "Why the time entry has been locked.",
            "nullable": true
          },
          "is_closed": {
            "type": "boolean",
            "description": "Whether or not the time entry has been approved via Timesheet Approval.",
            "nullable": true
          },
          "is_billed": {
            "type": "boolean",
            "description": "Whether or not the time entry has been marked as invoiced.",
            "nullable": true
          },
          "timer_started_at": {
            "type": "string",
            "description": "Date and time the running timer was started (if tracking by duration). Use the ISO 8601 Format. Returns null for stopped timers.",
            "nullable": true,
            "format": "date-time"
          },
          "started_time": {
            "type": "string",
            "description": "Time the time entry was started (if tracking by start/end times).",
            "nullable": true
          },
          "ended_time": {
            "type": "string",
            "description": "Time the time entry was ended (if tracking by start/end times).",
            "nullable": true
          },
          "is_running": {
            "type": "boolean",
            "description": "Whether or not the time entry is currently running.",
            "nullable": true
          },
          "billable": {
            "type": "boolean",
            "description": "Whether or not the time entry is billable.",
            "nullable": true
          },
          "budgeted": {
            "type": "boolean",
            "description": "Whether or not the time entry counts towards the project budget.",
            "nullable": true
          },
          "billable_rate": {
            "type": "number",
            "description": "The billable rate for the time entry.",
            "nullable": true,
            "format": "float"
          },
          "cost_rate": {
            "type": "number",
            "description": "The cost rate for the time entry.",
            "nullable": true,
            "format": "float"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the time entry was created. Use the ISO 8601 Format.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the time entry was last updated. Use the ISO 8601 Format.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "UserAssignment": {
        "type": "object",
        "externalDocs": {
          "description": "user-assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/user-assignments/#the-user-assignment-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the user assignment.",
            "nullable": true,
            "format": "int32"
          },
          "project": {
            "type": "object",
            "description": "An object containing the id, name, and code of the associated project.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "code": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "user": {
            "type": "object",
            "description": "An object containing the id and name of the associated user.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the user assignment is active or archived.",
            "nullable": true
          },
          "is_project_manager": {
            "type": "boolean",
            "description": "Determines if the user has Project Manager permissions for the project.",
            "nullable": true
          },
          "use_default_rates": {
            "type": "boolean",
            "description": "Determines which billable rate(s) will be used on the project for this user when bill_by is People. When true, the project will use the user’s default billable rates. When false, the project will use the custom rate defined on this user assignment.",
            "nullable": true
          },
          "hourly_rate": {
            "type": "number",
            "description": "Custom rate used when the project’s bill_by is People and use_default_rates is false.",
            "nullable": true,
            "format": "float"
          },
          "budget": {
            "type": "number",
            "description": "Budget used when the project’s budget_by is person.",
            "nullable": true,
            "format": "float"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the user assignment was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the user assignment was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "TaskAssignment": {
        "type": "object",
        "externalDocs": {
          "description": "task-assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/task-assignments/#the-task-assignment-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the task assignment.",
            "nullable": true,
            "format": "int32"
          },
          "project": {
            "type": "object",
            "description": "An object containing the id, name, and code of the associated project.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "code": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "task": {
            "type": "object",
            "description": "An object containing the id and name of the associated task.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the task assignment is active or archived.",
            "nullable": true
          },
          "billable": {
            "type": "boolean",
            "description": "Whether the task assignment is billable or not. For example: if set to true, all time tracked on this project for the associated task will be marked as billable.",
            "nullable": true
          },
          "hourly_rate": {
            "type": "number",
            "description": "Rate used when the project’s bill_by is Tasks.",
            "nullable": true,
            "format": "float"
          },
          "budget": {
            "type": "number",
            "description": "Budget used when the project’s budget_by is task or task_fees.",
            "nullable": true,
            "format": "float"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the task assignment was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the task assignment was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Project": {
        "type": "object",
        "externalDocs": {
          "description": "project",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/projects/#the-project-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the project.",
            "nullable": true,
            "format": "int32"
          },
          "client": {
            "type": "object",
            "description": "An object containing the project’s client id, name, and currency.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "currency": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "name": {
            "type": "string",
            "description": "Unique name for the project.",
            "nullable": true
          },
          "code": {
            "type": "string",
            "description": "The code associated with the project.",
            "nullable": true
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the project is active or archived.",
            "nullable": true
          },
          "is_billable": {
            "type": "boolean",
            "description": "Whether the project is billable or not.",
            "nullable": true
          },
          "is_fixed_fee": {
            "type": "boolean",
            "description": "Whether the project is a fixed-fee project or not.",
            "nullable": true
          },
          "bill_by": {
            "type": "string",
            "description": "The method by which the project is invoiced.",
            "nullable": true
          },
          "hourly_rate": {
            "type": "number",
            "description": "Rate for projects billed by Project Hourly Rate.",
            "nullable": true,
            "format": "float"
          },
          "budget": {
            "type": "number",
            "description": "The budget in hours for the project when budgeting by time.",
            "nullable": true,
            "format": "float"
          },
          "budget_by": {
            "type": "string",
            "description": "The method by which the project is budgeted.",
            "nullable": true
          },
          "budget_is_monthly": {
            "type": "boolean",
            "description": "Option to have the budget reset every month.",
            "nullable": true
          },
          "notify_when_over_budget": {
            "type": "boolean",
            "description": "Whether Project Managers should be notified when the project goes over budget.",
            "nullable": true
          },
          "over_budget_notification_percentage": {
            "type": "number",
            "description": "Percentage value used to trigger over budget email alerts.",
            "nullable": true,
            "format": "float"
          },
          "over_budget_notification_date": {
            "type": "string",
            "description": "Date of last over budget notification. If none have been sent, this will be null.",
            "nullable": true,
            "format": "date"
          },
          "show_budget_to_all": {
            "type": "boolean",
            "description": "Option to show project budget to all employees. Does not apply to Total Project Fee projects.",
            "nullable": true
          },
          "cost_budget": {
            "type": "number",
            "description": "The monetary budget for the project when budgeting by money.",
            "nullable": true,
            "format": "float"
          },
          "cost_budget_include_expenses": {
            "type": "boolean",
            "description": "Option for budget of Total Project Fees projects to include tracked expenses.",
            "nullable": true
          },
          "fee": {
            "type": "number",
            "description": "The amount you plan to invoice for the project. Only used by fixed-fee projects.",
            "nullable": true,
            "format": "float"
          },
          "notes": {
            "type": "string",
            "description": "Project notes.",
            "nullable": true
          },
          "starts_on": {
            "type": "string",
            "description": "Date the project was started.",
            "nullable": true,
            "format": "date"
          },
          "ends_on": {
            "type": "string",
            "description": "Date the project will end.",
            "nullable": true,
            "format": "date"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the project was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the project was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Role": {
        "type": "object",
        "externalDocs": {
          "description": "role",
          "url": "https://help.getharvest.com/api-v2/roles-api/roles/roles/#the-role-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the role.",
            "nullable": true,
            "format": "int32"
          },
          "name": {
            "type": "string",
            "description": "The name of the role.",
            "nullable": true
          },
          "user_ids": {
            "type": "array",
            "description": "The IDs of the users assigned to this role.",
            "nullable": true,
            "items": {
              "type": "integer"
            }
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the role was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the role was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "Teammate": {
        "type": "object",
        "externalDocs": {
          "description": "teammate",
          "url": "https://help.getharvest.com/api-v2/users-api/users/teammates/#the-teammate-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the teammate",
            "nullable": true
          },
          "first_name": {
            "type": "string",
            "description": "The first name of the teammate",
            "nullable": true
          },
          "last_name": {
            "type": "string",
            "description": "The last name of the teammate",
            "nullable": true
          },
          "email": {
            "type": "string",
            "description": "The email of the teammate",
            "nullable": true,
            "format": "email"
          }
        }
      },
      "BillableRate": {
        "type": "object",
        "externalDocs": {
          "description": "billable-rate",
          "url": "https://help.getharvest.com/api-v2/users-api/users/billable-rates/#the-billable-rate-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the billable rate.",
            "nullable": true,
            "format": "int32"
          },
          "amount": {
            "type": "number",
            "description": "The amount of the billable rate.",
            "nullable": true,
            "format": "float"
          },
          "start_date": {
            "type": "string",
            "description": "The date the billable rate is effective.",
            "nullable": true,
            "format": "date"
          },
          "end_date": {
            "type": "string",
            "description": "The date the billable rate is no longer effective. This date is calculated by Harvest.",
            "nullable": true,
            "format": "date"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the billable rate was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the billable rate was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "CostRate": {
        "type": "object",
        "externalDocs": {
          "description": "cost-rate",
          "url": "https://help.getharvest.com/api-v2/users-api/users/cost-rates/#the-cost-rate-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the cost rate.",
            "nullable": true,
            "format": "int32"
          },
          "amount": {
            "type": "number",
            "description": "The amount of the cost rate.",
            "nullable": true,
            "format": "float"
          },
          "start_date": {
            "type": "string",
            "description": "The date the cost rate is effective.",
            "nullable": true,
            "format": "date"
          },
          "end_date": {
            "type": "string",
            "description": "The date the cost rate is no longer effective. This date is calculated by Harvest.",
            "nullable": true,
            "format": "date"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the cost rate was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the cost rate was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "ProjectAssignment": {
        "type": "object",
        "externalDocs": {
          "description": "project-assignment",
          "url": "https://help.getharvest.com/api-v2/users-api/users/project-assignments/#the-project-assignment-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the project assignment.",
            "nullable": true,
            "format": "int32"
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the project assignment is active or archived.",
            "nullable": true
          },
          "is_project_manager": {
            "type": "boolean",
            "description": "Determines if the user has Project Manager permissions for the project.",
            "nullable": true
          },
          "use_default_rates": {
            "type": "boolean",
            "description": "Determines which billable rate(s) will be used on the project for this user when bill_by is People. When true, the project will use the user’s default billable rates. When false, the project will use the custom rate defined on this user assignment.",
            "nullable": true
          },
          "hourly_rate": {
            "type": "number",
            "description": "Custom rate used when the project’s bill_by is People and use_default_rates is false.",
            "nullable": true,
            "format": "float"
          },
          "budget": {
            "type": "number",
            "description": "Budget used when the project’s budget_by is person.",
            "nullable": true,
            "format": "float"
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the project assignment was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the project assignment was last updated.",
            "nullable": true,
            "format": "date-time"
          },
          "project": {
            "type": "object",
            "description": "An object containing the assigned project id, name, and code.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              },
              "code": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "client": {
            "type": "object",
            "description": "An object containing the project’s client id and name.",
            "nullable": true,
            "properties": {
              "id": {
                "type": "integer",
                "nullable": true
              },
              "name": {
                "type": "string",
                "nullable": true
              }
            }
          },
          "task_assignments": {
            "type": "array",
            "description": "Array of task assignment objects associated with the project.",
            "nullable": true,
            "items": {
              "$ref": "#/components/schemas/TaskAssignment"
            }
          }
        }
      },
      "User": {
        "type": "object",
        "externalDocs": {
          "description": "user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/users/#the-user-object"
        },
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique ID for the user.",
            "nullable": true,
            "format": "int32"
          },
          "first_name": {
            "type": "string",
            "description": "The first name of the user.",
            "nullable": true
          },
          "last_name": {
            "type": "string",
            "description": "The last name of the user.",
            "nullable": true
          },
          "email": {
            "type": "string",
            "description": "The email address of the user.",
            "nullable": true,
            "format": "email"
          },
          "telephone": {
            "type": "string",
            "description": "The user’s telephone number.",
            "nullable": true
          },
          "timezone": {
            "type": "string",
            "description": "The user’s timezone.",
            "nullable": true
          },
          "has_access_to_all_future_projects": {
            "type": "boolean",
            "description": "Whether the user should be automatically added to future projects.",
            "nullable": true
          },
          "is_contractor": {
            "type": "boolean",
            "description": "Whether the user is a contractor or an employee.",
            "nullable": true
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the user is active or archived.",
            "nullable": true
          },
          "weekly_capacity": {
            "type": "integer",
            "description": "The number of hours per week this person is available to work in seconds, in half hour increments. For example, if a person’s capacity is 35 hours, the API will return 126000 seconds.",
            "nullable": true,
            "format": "int32"
          },
          "default_hourly_rate": {
            "type": "number",
            "description": "The billable rate to use for this user when they are added to a project.",
            "nullable": true,
            "format": "float"
          },
          "cost_rate": {
            "type": "number",
            "description": "The cost rate to use for this user when calculating a project’s costs vs billable amount.",
            "nullable": true,
            "format": "float"
          },
          "roles": {
            "type": "array",
            "description": "Descriptive names of the business roles assigned to this person. They can be used for filtering reports, and have no effect in their permissions in Harvest.",
            "nullable": true,
            "items": {
              "type": "string"
            }
          },
          "access_roles": {
            "type": "array",
            "description": "Access role(s) that determine the user’s permissions in Harvest. Possible values: administrator, manager or member. Users with the manager role can additionally be granted one or more of these roles: project_creator, billable_rates_manager, managed_projects_invoice_drafter, managed_projects_invoice_manager, client_and_task_manager, time_and_expenses_manager, estimates_manager.",
            "nullable": true,
            "items": {
              "type": "string"
            }
          },
          "avatar_url": {
            "type": "string",
            "description": "The URL to the user’s avatar image.",
            "nullable": true
          },
          "created_at": {
            "type": "string",
            "description": "Date and time the user was created.",
            "nullable": true,
            "format": "date-time"
          },
          "updated_at": {
            "type": "string",
            "description": "Date and time the user was last updated.",
            "nullable": true,
            "format": "date-time"
          }
        }
      },
      "ExpenseReportsResult": {
        "type": "object",
        "externalDocs": {
          "description": "result",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/expense-reports/#the-result-object"
        },
        "properties": {
          "client_id": {
            "type": "integer",
            "description": "The ID of the client associated with the reported expenses. Only returned in the Client and Project reports.",
            "nullable": true,
            "format": "int32"
          },
          "client_name": {
            "type": "string",
            "description": "The name of the client associated with the reported expenses. Only returned in the Client and Project reports.",
            "nullable": true
          },
          "project_id": {
            "type": "integer",
            "description": "The ID of the project associated with the reported expenses. Only returned in the Client and Project reports.",
            "nullable": true,
            "format": "int32"
          },
          "project_name": {
            "type": "string",
            "description": "The name of the project associated with the reported expenses. Only returned in the Client and Project reports.",
            "nullable": true
          },
          "expense_category_id": {
            "type": "integer",
            "description": "The ID of the expense category associated with the reported expenses. Only returned in the Expense Category report.",
            "nullable": true,
            "format": "int32"
          },
          "expense_category_name": {
            "type": "string",
            "description": "The name of the expense category associated with the reported expenses. Only returned in the Expense Category report.",
            "nullable": true
          },
          "user_id": {
            "type": "integer",
            "description": "The ID of the user associated with the reported expenses. Only returned in the Team report.",
            "nullable": true,
            "format": "int32"
          },
          "user_name": {
            "type": "string",
            "description": "The name of the user associated with the reported expenses. Only returned in the Team report.",
            "nullable": true
          },
          "is_contractor": {
            "type": "boolean",
            "description": "The contractor status of the user associated with the reported expenses. Only returned in the Team report.",
            "nullable": true
          },
          "total_amount": {
            "type": "number",
            "description": "The totaled cost for all expenses for the given timeframe, subject (client, project, expense category, or user), and currency.",
            "nullable": true,
            "format": "float"
          },
          "billable_amount": {
            "type": "number",
            "description": "The totaled cost for billable expenses for the given timeframe, subject (client, project, expense category, or user), and currency.",
            "nullable": true,
            "format": "float"
          },
          "currency": {
            "type": "string",
            "description": "The currency code associated with the expenses for this result.",
            "nullable": true
          }
        }
      },
      "UninvoicedReportResult": {
        "type": "object",
        "externalDocs": {
          "description": "result",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/uninvoiced-report/#the-result-object"
        },
        "properties": {
          "client_id": {
            "type": "integer",
            "description": "The ID of the client associated with the reported hours and expenses.",
            "nullable": true,
            "format": "int32"
          },
          "client_name": {
            "type": "string",
            "description": "The name of the client associated with the reported hours and expenses.",
            "nullable": true
          },
          "project_id": {
            "type": "integer",
            "description": "The ID of the project associated with the reported hours and expenses.",
            "nullable": true,
            "format": "int32"
          },
          "project_name": {
            "type": "string",
            "description": "The name of the project associated with the reported hours and expenses.",
            "nullable": true
          },
          "currency": {
            "type": "string",
            "description": "The currency code associated with the tracked hours for this result.",
            "nullable": true
          },
          "total_hours": {
            "type": "number",
            "description": "The total hours for the given timeframe and project. If Time Rounding is turned on, the hours will be rounded according to your settings.",
            "nullable": true,
            "format": "float"
          },
          "uninvoiced_hours": {
            "type": "number",
            "description": "The total hours for the given timeframe and project that have not been invoiced. If Time Rounding is turned on, the hours will be rounded according to your settings.",
            "nullable": true,
            "format": "float"
          },
          "uninvoiced_expenses": {
            "type": "number",
            "description": "The total amount for billable expenses for the timeframe and project that have not been invoiced.",
            "nullable": true,
            "format": "float"
          },
          "uninvoiced_amount": {
            "type": "number",
            "description": "The total amount (time and expenses) for the timeframe and project that have not been invoiced.",
            "nullable": true,
            "format": "float"
          }
        }
      },
      "TimeReportsResult": {
        "type": "object",
        "externalDocs": {
          "description": "result",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/time-reports/#the-result-object"
        },
        "properties": {
          "client_id": {
            "type": "integer",
            "description": "The ID of the client associated with the reported hours. Only returned in the Client and Project reports.",
            "nullable": true,
            "format": "int32"
          },
          "client_name": {
            "type": "string",
            "description": "The name of the client associated with the reported hours. Only returned in the Client and Project reports.",
            "nullable": true
          },
          "project_id": {
            "type": "integer",
            "description": "The ID of the project associated with the reported hours. Only returned in the Client and Project reports.",
            "nullable": true,
            "format": "int32"
          },
          "project_name": {
            "type": "string",
            "description": "The name of the project associated with the reported hours. Only returned in the Client and Project reports.",
            "nullable": true
          },
          "task_id": {
            "type": "integer",
            "description": "The ID of the task associated with the reported hours. Only returned in the Task report.",
            "nullable": true,
            "format": "int32"
          },
          "task_name": {
            "type": "string",
            "description": "The name of the task associated with the reported hours. Only returned in the Task report.",
            "nullable": true
          },
          "user_id": {
            "type": "integer",
            "description": "The ID of the user associated with the reported hours. Only returned in the Team report.",
            "nullable": true,
            "format": "int32"
          },
          "user_name": {
            "type": "string",
            "description": "The name of the user associated with the reported hours. Only returned in the Team report.",
            "nullable": true
          },
          "weekly_capacity": {
            "type": "integer",
            "description": "The number of hours per week this person is available to work in seconds, in half hour increments. For example, if a person’s capacity is 35 hours, the API will return 126000 seconds. Only returned in the Team report.",
            "nullable": true,
            "format": "int32"
          },
          "avatar_url": {
            "type": "string",
            "description": "The URL to the user’s avatar image. Only returned in the Team report.",
            "nullable": true
          },
          "is_contractor": {
            "type": "boolean",
            "description": "The contractor status of the user associated with the reported hours. Only returned in the Team report.",
            "nullable": true
          },
          "total_hours": {
            "type": "number",
            "description": "The totaled hours for the given timeframe, subject (client, project, task, or user), and currency. If Time Rounding is turned on, the hours will be rounded according to your settings.",
            "nullable": true,
            "format": "float"
          },
          "billable_hours": {
            "type": "number",
            "description": "The totaled billable hours for the given timeframe, subject (client, project, task, or user), and currency. If Time Rounding is turned on, the hours will be rounded according to your settings.",
            "nullable": true,
            "format": "float"
          },
          "currency": {
            "type": "string",
            "description": "The currency code associated with the tracked hours for this result. Only visible to Administrators and Project Managers with the View billable rates and amounts permission.",
            "nullable": true
          },
          "billable_amount": {
            "type": "number",
            "description": "The totaled billable amount for the billable hours above. Only visible to Administrators and Project Managers with the View billable rates and amounts permission.",
            "nullable": true,
            "format": "float"
          }
        }
      },
      "ProjectBudgetReportResult": {
        "type": "object",
        "externalDocs": {
          "description": "result",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/project-budget-report/#the-result-object"
        },
        "properties": {
          "client_id": {
            "type": "integer",
            "description": "The ID of the client associated with this project.",
            "nullable": true,
            "format": "int32"
          },
          "client_name": {
            "type": "string",
            "description": "The name of the client associated with this project.",
            "nullable": true
          },
          "project_id": {
            "type": "integer",
            "description": "The ID of the project.",
            "nullable": true,
            "format": "int32"
          },
          "project_name": {
            "type": "string",
            "description": "The name of the project.",
            "nullable": true
          },
          "budget_is_monthly": {
            "type": "boolean",
            "description": "Whether the budget is reset every month.",
            "nullable": true
          },
          "budget_by": {
            "type": "string",
            "description": "The method by which the project is budgeted. Options: project (Hours Per Project), project_cost (Total Project Fees), task (Hours Per Task), task_fees (Fees Per Task), person (Hours Per Person), none (No Budget).",
            "nullable": true
          },
          "is_active": {
            "type": "boolean",
            "description": "Whether the project is active or archived.",
            "nullable": true
          },
          "budget": {
            "type": "number",
            "description": "The budget in hours or money for the project when budgeting by time. If the project is budgeted by money, this value will only be visible to Administrators and Project Managers with the View billable rates and amounts permission.",
            "nullable": true,
            "format": "float"
          },
          "budget_spent": {
            "type": "number",
            "description": "The total hours or money spent against the project’s budget. If Time Rounding is turned on, the hours will be rounded according to your settings. If the project is budgeted by money, this value will only be visible to Administrators and Project Managers with the View billable rates and amounts permission.",
            "nullable": true,
            "format": "float"
          },
          "budget_remaining": {
            "type": "number",
            "description": "The total hours or money remaining in the project’s budget. If Time Rounding is turned on, the hours will be rounded according to your settings. If the project is budgeted by money, this value will only be visible to Administrators and Project Managers with the View billable rates and amounts permission.",
            "nullable": true,
            "format": "float"
          }
        }
      },
      "Contacts": {
        "type": "object",
        "required": [
          "contacts",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "contacts": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Contact"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Clients": {
        "type": "object",
        "required": [
          "clients",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "clients": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Client"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Companies": {
        "type": "object",
        "required": [
          "companies",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "companies": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Company"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "InvoiceMessages": {
        "type": "object",
        "required": [
          "invoice_messages",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "invoice_messages": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/InvoiceMessage"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "InvoiceMessageRecipients": {
        "type": "object",
        "required": [
          "invoice_message_recipients",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "invoice_message_recipients": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/InvoiceMessageRecipient"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "InvoicePayments": {
        "type": "object",
        "required": [
          "invoice_payments",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "invoice_payments": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/InvoicePayment"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Invoices": {
        "type": "object",
        "required": [
          "invoices",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "invoices": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Invoice"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "InvoiceLineItems": {
        "type": "object",
        "required": [
          "invoice_line_items",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "invoice_line_items": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/InvoiceLineItem"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "InvoiceItemCategories": {
        "type": "object",
        "required": [
          "invoice_item_categories",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "invoice_item_categories": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/InvoiceItemCategory"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "EstimateMessages": {
        "type": "object",
        "required": [
          "estimate_messages",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "estimate_messages": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/EstimateMessage"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "EstimateMessageRecipients": {
        "type": "object",
        "required": [
          "estimate_message_recipients",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "estimate_message_recipients": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/EstimateMessageRecipient"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Estimates": {
        "type": "object",
        "required": [
          "estimates",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "estimates": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Estimate"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "EstimateLineItems": {
        "type": "object",
        "required": [
          "estimate_line_items",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "estimate_line_items": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/EstimateLineItem"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "EstimateItemCategories": {
        "type": "object",
        "required": [
          "estimate_item_categories",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "estimate_item_categories": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/EstimateItemCategory"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Expenses": {
        "type": "object",
        "required": [
          "expenses",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "expenses": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Expense"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "ExpenseCategories": {
        "type": "object",
        "required": [
          "expense_categories",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "expense_categories": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ExpenseCategory"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Tasks": {
        "type": "object",
        "required": [
          "tasks",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "tasks": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Task"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "TimeEntries": {
        "type": "object",
        "required": [
          "time_entries",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "time_entries": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/TimeEntry"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "UserAssignments": {
        "type": "object",
        "required": [
          "user_assignments",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "user_assignments": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/UserAssignment"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "TaskAssignments": {
        "type": "object",
        "required": [
          "task_assignments",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "task_assignments": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/TaskAssignment"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Projects": {
        "type": "object",
        "required": [
          "projects",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "projects": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Project"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Roles": {
        "type": "object",
        "required": [
          "roles",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "roles": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Role"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Teammates": {
        "type": "object",
        "required": [
          "teammates",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "teammates": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Teammate"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "BillableRates": {
        "type": "object",
        "required": [
          "billable_rates",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "billable_rates": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/BillableRate"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "CostRates": {
        "type": "object",
        "required": [
          "cost_rates",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "cost_rates": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/CostRate"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "ProjectAssignments": {
        "type": "object",
        "required": [
          "project_assignments",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "project_assignments": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ProjectAssignment"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Users": {
        "type": "object",
        "required": [
          "users",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "users": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/User"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "ExpenseReportsResults": {
        "type": "object",
        "required": [
          "results",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "results": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ExpenseReportsResult"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "UninvoicedReportResults": {
        "type": "object",
        "required": [
          "results",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "results": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/UninvoicedReportResult"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "TimeReportsResults": {
        "type": "object",
        "required": [
          "results",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "results": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/TimeReportsResult"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "ProjectBudgetReportResults": {
        "type": "object",
        "required": [
          "results",
          "per_page",
          "total_pages",
          "total_entries",
          "next_page",
          "previous_page",
          "page",
          "links"
        ],
        "properties": {
          "results": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ProjectBudgetReportResult"
            }
          },
          "per_page": {
            "type": "integer",
            "format": "int64"
          },
          "total_pages": {
            "type": "integer",
            "format": "int64"
          },
          "total_entries": {
            "type": "integer",
            "format": "int64"
          },
          "next_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "previous_page": {
            "type": "integer",
            "format": "int64",
            "nullable": true
          },
          "page": {
            "type": "integer",
            "format": "int64"
          },
          "links": {
            "$ref": "#/components/schemas/PaginationLinks"
          }
        }
      },
      "Error": {
        "type": "object",
        "properties": {
          "code": {
            "type": "integer"
          },
          "message": {
            "type": "string"
          }
        }
      },
      "InvoiceMessageSubjectAndBody": {
        "type": "object",
        "required": [
          "invoice_id",
          "subject",
          "body",
          "reminder",
          "thank_you"
        ],
        "properties": {
          "invoice_id": {
            "type": "integer",
            "format": "int32"
          },
          "subject": {
            "type": "string"
          },
          "body": {
            "type": "string"
          },
          "reminder": {
            "type": "boolean"
          },
          "thank_you": {
            "type": "boolean"
          }
        }
      },
      "PaginationLinks": {
        "type": "object",
        "required": [
          "first",
          "last"
        ],
        "properties": {
          "first": {
            "type": "string",
            "format": "url",
            "description": "First page"
          },
          "last": {
            "type": "string",
            "format": "url",
            "description": "Last page"
          },
          "previous": {
            "type": "string",
            "format": "url",
            "description": "Previous page",
            "nullable": true
          },
          "next": {
            "type": "string",
            "format": "url",
            "description": "Next page",
            "nullable": true
          }
        }
      },
      "TeammatesPatchResponse": {
        "type": "object",
        "required": [
          "teammates"
        ],
        "properties": {
          "teammates": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Teammate"
            }
          }
        }
      }
    }
  },
  "security": [
    {
      "BearerAuth": [],
      "AccountAuth": []
    }
  ],
  "paths": {
    "/clients": {
      "get": {
        "summary": "List all clients",
        "operationId": "listClients",
        "description": "Returns a list of your clients. The clients are returned sorted by creation date, with the most recently created clients appearing first.\n\nThe response contains an object with a clients property that contains an array of up to per_page clients. Each entry in the array is a separate client object. If no more clients are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your clients.",
        "externalDocs": {
          "description": "List all clients",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/clients/#list-all-clients"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all clients",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Clients"
                },
                "example": {
                  "clients": [
                    {
                      "id": 5735776,
                      "name": "123 Industries",
                      "is_active": true,
                      "address": "123 Main St.\r\nAnytown, LA 71223",
                      "statement_key": "0a39d3e33c8058cf7c3f8097d854c64e",
                      "created_at": "2017-06-26T21:02:12Z",
                      "updated_at": "2017-06-26T21:34:11Z",
                      "currency": "EUR"
                    },
                    {
                      "id": 5735774,
                      "name": "ABC Corp",
                      "is_active": true,
                      "address": "456 Main St.\r\nAnytown, CT 06467",
                      "statement_key": "e42aa2cb60e85925ffe5d13ee7ee8706",
                      "created_at": "2017-06-26T21:01:52Z",
                      "updated_at": "2017-06-26T21:27:07Z",
                      "currency": "USD"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/clients?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/clients?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "is_active",
            "description": "Pass true to only return active clients and false to return inactive clients.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return clients that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a client",
        "operationId": "createClient",
        "description": "Creates a new client object. Returns a client object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a client",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/clients/#create-a-client"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a client",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Client"
                },
                "example": {
                  "id": 5737336,
                  "name": "Your New Client",
                  "is_active": true,
                  "address": null,
                  "statement_key": "82455699ad085d8cffc3e9a4e43ff7b8",
                  "created_at": "2017-06-26T21:39:35Z",
                  "updated_at": "2017-06-26T21:39:35Z",
                  "currency": "EUR"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "A textual description of the client.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the client is active, or archived. Defaults to true.",
                    "nullable": true
                  },
                  "address": {
                    "type": "string",
                    "description": "A textual representation of the client’s physical address. May include new line characters.",
                    "nullable": true
                  },
                  "currency": {
                    "type": "string",
                    "description": "The currency used by the client. If not provided, the company’s currency will be used. See a list of supported currencies",
                    "nullable": true
                  }
                },
                "required": [
                  "name"
                ]
              }
            }
          }
        }
      }
    },
    "/clients/{clientId}": {
      "delete": {
        "summary": "Delete a client",
        "operationId": "deleteClient",
        "description": "Delete a client. Deleting a client is only possible if it has no projects, invoices, or estimates associated with it. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a client",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/clients/#delete-a-client"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a client"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "clientId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a client",
        "operationId": "retrieveClient",
        "description": "Retrieves the client with the given ID. Returns a client object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a client",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/clients/#retrieve-a-client"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a client",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Client"
                },
                "example": {
                  "id": 5735776,
                  "name": "123 Industries",
                  "is_active": true,
                  "address": "123 Main St.\r\nAnytown, LA 71223",
                  "statement_key": "0a39d3e33c8058cf7c3f8097d854c64e",
                  "created_at": "2017-06-26T21:02:12Z",
                  "updated_at": "2017-06-26T21:34:11Z",
                  "currency": "EUR"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "clientId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a client",
        "operationId": "updateClient",
        "description": "Updates the specific client by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a client object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a client",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/clients/#update-a-client"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a client",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Client"
                },
                "example": {
                  "id": 5737336,
                  "name": "Your New Client",
                  "is_active": false,
                  "address": null,
                  "statement_key": "82455699ad085d8cffc3e9a4e43ff7b8",
                  "created_at": "2017-06-26T21:39:35Z",
                  "updated_at": "2017-06-26T21:41:18Z",
                  "currency": "EUR"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "clientId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "A textual description of the client.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the client is active, or archived.",
                    "nullable": true
                  },
                  "address": {
                    "type": "string",
                    "description": "A textual representation of the client’s physical address. May include new line characters.",
                    "nullable": true
                  },
                  "currency": {
                    "type": "string",
                    "description": "The currency used by the client. See a list of supported currencies",
                    "nullable": true
                  }
                }
              }
            }
          }
        }
      }
    },
    "/company": {
      "get": {
        "summary": "Retrieve a company",
        "operationId": "retrieveCompany",
        "description": "Retrieves the company for the currently authenticated user. Returns a\ncompany object and a 200 OK response code.",
        "externalDocs": {
          "description": "Retrieve a company",
          "url": "https://help.getharvest.com/api-v2/company-api/company/company/#retrieve-a-company"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a company",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Company"
                },
                "example": {
                  "base_uri": "https://{ACCOUNT_SUBDOMAIN}.harvestapp.com",
                  "full_domain": "{ACCOUNT_SUBDOMAIN}.harvestapp.com",
                  "name": "API Examples",
                  "is_active": true,
                  "week_start_day": "Monday",
                  "wants_timestamp_timers": true,
                  "time_format": "hours_minutes",
                  "date_format": "%Y-%m-%d",
                  "plan_type": "sponsored",
                  "expense_feature": true,
                  "invoice_feature": true,
                  "estimate_feature": true,
                  "approval_feature": true,
                  "clock": "12h",
                  "currency_code_display": "iso_code_none",
                  "currency_symbol_display": "symbol_before",
                  "decimal_symbol": ".",
                  "thousands_separator": ",",
                  "color_scheme": "orange",
                  "weekly_capacity": 126000
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "patch": {
        "summary": "Update a company",
        "operationId": "updateCompany",
        "description": "Updates the company setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a company object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a company",
          "url": "https://help.getharvest.com/api-v2/company-api/company/company/#update-a-company"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a company",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Company"
                },
                "example": {
                  "base_uri": "https://{ACCOUNT_SUBDOMAIN}.harvestapp.com",
                  "full_domain": "{ACCOUNT_SUBDOMAIN}.harvestapp.com",
                  "name": "API Examples",
                  "is_active": true,
                  "week_start_day": "Monday",
                  "wants_timestamp_timers": false,
                  "time_format": "hours_minutes",
                  "date_format": "%Y-%m-%d",
                  "plan_type": "sponsored",
                  "expense_feature": true,
                  "invoice_feature": true,
                  "estimate_feature": true,
                  "approval_feature": true,
                  "clock": "12h",
                  "currency_code_display": "iso_code_none",
                  "currency_symbol_display": "symbol_before",
                  "decimal_symbol": ".",
                  "thousands_separator": ",",
                  "color_scheme": "orange",
                  "weekly_capacity": 108000
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "wants_timestamp_timers": {
                    "type": "boolean",
                    "description": "Whether time is tracked via duration or start and end times.",
                    "nullable": true
                  },
                  "weekly_capacity": {
                    "type": "integer",
                    "description": "The weekly capacity in seconds.",
                    "nullable": true,
                    "format": "int32"
                  }
                }
              }
            }
          }
        }
      }
    },
    "/contacts": {
      "get": {
        "summary": "List all contacts",
        "operationId": "listContacts",
        "description": "Returns a list of your contacts. The contacts are returned sorted by creation date, with the most recently created contacts appearing first.\n\nThe response contains an object with a contacts property that contains an array of up to per_page contacts. Each entry in the array is a separate contact object. If no more contacts are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your contacts.",
        "externalDocs": {
          "description": "List all contacts",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/contacts/#list-all-contacts"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all contacts",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Contacts"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "client_id",
            "description": "Only return contacts belonging to the client with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return contacts that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a contact",
        "operationId": "createContact",
        "description": "Creates a new contact object. Returns a contact object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a contact",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/contacts/#create-a-contact"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a contact",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Contact"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client associated with this contact.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "title": {
                    "type": "string",
                    "description": "The title of the contact.",
                    "nullable": true
                  },
                  "first_name": {
                    "type": "string",
                    "description": "The first name of the contact.",
                    "nullable": true
                  },
                  "last_name": {
                    "type": "string",
                    "description": "The last name of the contact.",
                    "nullable": true
                  },
                  "email": {
                    "type": "string",
                    "description": "The contact’s email address.",
                    "nullable": true,
                    "format": "email"
                  },
                  "phone_office": {
                    "type": "string",
                    "description": "The contact’s office phone number.",
                    "nullable": true
                  },
                  "phone_mobile": {
                    "type": "string",
                    "description": "The contact’s mobile phone number.",
                    "nullable": true
                  },
                  "fax": {
                    "type": "string",
                    "description": "The contact’s fax number.",
                    "nullable": true
                  }
                },
                "required": [
                  "client_id",
                  "first_name"
                ]
              }
            }
          }
        }
      }
    },
    "/contacts/{contactId}": {
      "delete": {
        "summary": "Delete a contact",
        "operationId": "deleteContact",
        "description": "Delete a contact. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a contact",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/contacts/#delete-a-contact"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a contact"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "contactId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a contact",
        "operationId": "retrieveContact",
        "description": "Retrieves the contact with the given ID. Returns a contact object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a contact",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/contacts/#retrieve-a-contact"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a contact",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Contact"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "contactId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a contact",
        "operationId": "updateContact",
        "description": "Updates the specific contact by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a contact object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a contact",
          "url": "https://help.getharvest.com/api-v2/clients-api/clients/contacts/#update-a-contact"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a contact",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Contact"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "contactId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client associated with this contact.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "title": {
                    "type": "string",
                    "description": "The title of the contact.",
                    "nullable": true
                  },
                  "first_name": {
                    "type": "string",
                    "description": "The first name of the contact.",
                    "nullable": true
                  },
                  "last_name": {
                    "type": "string",
                    "description": "The last name of the contact.",
                    "nullable": true
                  },
                  "email": {
                    "type": "string",
                    "description": "The contact’s email address.",
                    "nullable": true,
                    "format": "email"
                  },
                  "phone_office": {
                    "type": "string",
                    "description": "The contact’s office phone number.",
                    "nullable": true
                  },
                  "phone_mobile": {
                    "type": "string",
                    "description": "The contact’s mobile phone number.",
                    "nullable": true
                  },
                  "fax": {
                    "type": "string",
                    "description": "The contact’s fax number.",
                    "nullable": true
                  }
                }
              }
            }
          }
        }
      }
    },
    "/estimate_item_categories": {
      "get": {
        "summary": "List all estimate item categories",
        "operationId": "listEstimateItemCategories",
        "description": "Returns a list of your estimate item categories. The estimate item categories are returned sorted by creation date, with the most recently created estimate item categories appearing first.\n\nThe response contains an object with a estimate_item_categories property that contains an array of up to per_page estimate item categories. Each entry in the array is a separate estimate item category object. If no more estimate item categories are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your estimate item categories.",
        "externalDocs": {
          "description": "List all estimate item categories",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-item-categories/#list-all-estimate-item-categories"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all estimate item categories",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/EstimateItemCategories"
                },
                "example": {
                  "estimate_item_categories": [
                    {
                      "id": 1378704,
                      "name": "Product",
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T20:41:00Z"
                    },
                    {
                      "id": 1378703,
                      "name": "Service",
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T20:41:00Z"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/estimate_item_categories?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/estimate_item_categories?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "updated_since",
            "description": "Only return estimate item categories that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an estimate item category",
        "operationId": "createEstimateItemCategory",
        "description": "Creates a new estimate item category object. Returns an estimate item category object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an estimate item category",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-item-categories/#create-an-estimate-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an estimate item category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/EstimateItemCategory"
                },
                "example": {
                  "id": 1379244,
                  "name": "Hosting",
                  "created_at": "2017-06-27T16:06:35Z",
                  "updated_at": "2017-06-27T16:06:35Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the estimate item category.",
                    "nullable": true
                  }
                },
                "required": [
                  "name"
                ]
              }
            }
          }
        }
      }
    },
    "/estimate_item_categories/{estimateItemCategoryId}": {
      "delete": {
        "summary": "Delete an estimate item category",
        "operationId": "deleteEstimateItemCategory",
        "description": "Delete an estimate item category. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an estimate item category",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-item-categories/#delete-an-estimate-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an estimate item category"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateItemCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve an estimate item category",
        "operationId": "retrieveEstimateItemCategory",
        "description": "Retrieves the estimate item category with the given ID. Returns an estimate item category object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve an estimate item category",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-item-categories/#retrieve-an-estimate-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve an estimate item category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/EstimateItemCategory"
                },
                "example": {
                  "id": 1378704,
                  "name": "Product",
                  "created_at": "2017-06-26T20:41:00Z",
                  "updated_at": "2017-06-26T20:41:00Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateItemCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update an estimate item category",
        "operationId": "updateEstimateItemCategory",
        "description": "Updates the specific estimate item category by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns an estimate item category object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update an estimate item category",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-item-categories/#update-an-estimate-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update an estimate item category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/EstimateItemCategory"
                },
                "example": {
                  "id": 1379244,
                  "name": "Transportation",
                  "created_at": "2017-06-27T16:06:35Z",
                  "updated_at": "2017-06-27T16:07:05Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateItemCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the estimate item category.",
                    "nullable": true
                  }
                }
              }
            }
          }
        }
      }
    },
    "/estimates": {
      "get": {
        "summary": "List all estimates",
        "operationId": "listEstimates",
        "description": "Returns a list of your estimates. The estimates are returned sorted by issue date, with the most recently issued estimates appearing first.\n\nThe response contains an object with a estimates property that contains an array of up to per_page estimates. Each entry in the array is a separate estimate object. If no more estimates are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your estimates.",
        "externalDocs": {
          "description": "List all estimates",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimates/#list-all-estimates"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all estimates",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Estimates"
                },
                "example": {
                  "estimates": [
                    {
                      "id": 1439818,
                      "client_key": "13dc088aa7d51ec687f186b146730c3c75dc7423",
                      "number": "1001",
                      "purchase_order": "5678",
                      "amount": 9630,
                      "tax": 5,
                      "tax_amount": 450,
                      "tax2": 2,
                      "tax2_amount": 180,
                      "discount": 10,
                      "discount_amount": 1000,
                      "subject": "Online Store - Phase 2",
                      "notes": "Some notes about the estimate",
                      "state": "sent",
                      "issue_date": "2017-06-01",
                      "sent_at": "2017-06-27T16:11:33Z",
                      "created_at": "2017-06-27T16:11:24Z",
                      "updated_at": "2017-06-27T16:13:56Z",
                      "accepted_at": null,
                      "declined_at": null,
                      "currency": "USD",
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "creator": {
                        "id": 1782884,
                        "name": "Bob Powell"
                      },
                      "line_items": [
                        {
                          "id": 53334195,
                          "kind": "Service",
                          "description": "Phase 2 of the Online Store",
                          "quantity": 100,
                          "unit_price": 100,
                          "amount": 10000,
                          "taxed": true,
                          "taxed2": true
                        }
                      ]
                    },
                    {
                      "id": 1439814,
                      "client_key": "a5ffaeb30c55776270fcd3992b70332d769f97e7",
                      "number": "1000",
                      "purchase_order": "1234",
                      "amount": 21000,
                      "tax": 5,
                      "tax_amount": 1000,
                      "tax2": null,
                      "tax2_amount": 0,
                      "discount": null,
                      "discount_amount": 0,
                      "subject": "Online Store - Phase 1",
                      "notes": "Some notes about the estimate",
                      "state": "accepted",
                      "issue_date": "2017-01-01",
                      "sent_at": "2017-06-27T16:10:30Z",
                      "created_at": "2017-06-27T16:09:33Z",
                      "updated_at": "2017-06-27T16:12:00Z",
                      "accepted_at": "2017-06-27T16:10:32Z",
                      "declined_at": null,
                      "currency": "USD",
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "creator": {
                        "id": 1782884,
                        "name": "Bob Powell"
                      },
                      "line_items": [
                        {
                          "id": 57531966,
                          "kind": "Service",
                          "description": "Phase 1 of the Online Store",
                          "quantity": 1,
                          "unit_price": 20000,
                          "amount": 20000,
                          "taxed": true,
                          "taxed2": false
                        }
                      ]
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/estimates?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/estimates?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "client_id",
            "description": "Only return estimates belonging to the client with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return estimates that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "from",
            "description": "Only return estimates with an issue_date on or after the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only return estimates with an issue_date on or before the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "state",
            "description": "Only return estimates with a state matching the value provided. Options: draft, sent, accepted, or declined.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an estimate",
        "operationId": "createEstimate",
        "description": "Creates a new estimate object. Returns an estimate object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an estimate",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimates/#create-an-estimate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an estimate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Estimate"
                },
                "example": {
                  "id": 1439827,
                  "client_key": "ddd4504a68fb7339138d0c2ea89ba05a3cf12aa8",
                  "number": "1002",
                  "purchase_order": null,
                  "amount": 5000,
                  "tax": null,
                  "tax_amount": 0,
                  "tax2": null,
                  "tax2_amount": 0,
                  "discount": null,
                  "discount_amount": 0,
                  "subject": "Project Quote",
                  "notes": null,
                  "state": "draft",
                  "issue_date": null,
                  "sent_at": null,
                  "created_at": "2017-06-27T16:16:24Z",
                  "updated_at": "2017-06-27T16:16:24Z",
                  "accepted_at": null,
                  "declined_at": null,
                  "currency": "USD",
                  "client": {
                    "id": 5735774,
                    "name": "ABC Corp"
                  },
                  "creator": {
                    "id": 1782884,
                    "name": "Bob Powell"
                  },
                  "line_items": [
                    {
                      "id": 53339199,
                      "kind": "Service",
                      "description": "Project Description",
                      "quantity": 1,
                      "unit_price": 5000,
                      "amount": 5000,
                      "taxed": false,
                      "taxed2": false
                    }
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client this estimate belongs to.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "number": {
                    "type": "string",
                    "description": "If no value is set, the number will be automatically generated.",
                    "nullable": true
                  },
                  "purchase_order": {
                    "type": "string",
                    "description": "The purchase order number.",
                    "nullable": true
                  },
                  "tax": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "tax2": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "discount": {
                    "type": "number",
                    "description": "This percentage is subtracted from the subtotal. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "subject": {
                    "type": "string",
                    "description": "The estimate subject.",
                    "nullable": true
                  },
                  "notes": {
                    "type": "string",
                    "description": "Any additional notes to include on the estimate.",
                    "nullable": true
                  },
                  "currency": {
                    "type": "string",
                    "description": "The currency used by the estimate. If not provided, the client’s currency will be used. See a list of supported currencies",
                    "nullable": true
                  },
                  "issue_date": {
                    "type": "string",
                    "description": "Date the estimate was issued. Defaults to today’s date.",
                    "nullable": true,
                    "format": "date"
                  },
                  "line_items": {
                    "type": "array",
                    "description": "Array of line item parameters",
                    "nullable": true,
                    "items": {
                      "type": "object",
                      "required": [
                        "kind",
                        "unit_price"
                      ],
                      "properties": {
                        "kind": {
                          "description": "The name of an estimate item category.",
                          "type": "string"
                        },
                        "description": {
                          "description": "Text description of the line item.",
                          "type": "string"
                        },
                        "quantity": {
                          "description": "The unit quantity of the item. Defaults to 1.",
                          "type": "integer",
                          "format": "int32"
                        },
                        "unit_price": {
                          "description": "The individual price per unit.",
                          "type": "number",
                          "format": "float"
                        },
                        "taxed": {
                          "description": "Whether the estimate’s tax percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        },
                        "taxed2": {
                          "description": "Whether the estimate’s tax2 percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        }
                      }
                    }
                  }
                },
                "required": [
                  "client_id"
                ]
              }
            }
          }
        }
      }
    },
    "/estimates/{estimateId}": {
      "delete": {
        "summary": "Delete an estimate",
        "operationId": "deleteEstimate",
        "description": "Delete an estimate. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an estimate",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimates/#delete-an-estimate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an estimate"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve an estimate",
        "operationId": "retrieveEstimate",
        "description": "Retrieves the estimate with the given ID. Returns an estimate object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve an estimate",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimates/#retrieve-an-estimate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve an estimate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Estimate"
                },
                "example": {
                  "id": 1439818,
                  "client_key": "13dc088aa7d51ec687f186b146730c3c75dc7423",
                  "number": "1001",
                  "purchase_order": "5678",
                  "amount": 9630,
                  "tax": 5,
                  "tax_amount": 450,
                  "tax2": 2,
                  "tax2_amount": 180,
                  "discount": 10,
                  "discount_amount": 1000,
                  "subject": "Online Store - Phase 2",
                  "notes": "Some notes about the estimate",
                  "state": "sent",
                  "issue_date": "2017-06-01",
                  "sent_at": "2017-06-27T16:11:33Z",
                  "created_at": "2017-06-27T16:11:24Z",
                  "updated_at": "2017-06-27T16:13:56Z",
                  "accepted_at": null,
                  "declined_at": null,
                  "currency": "USD",
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries"
                  },
                  "creator": {
                    "id": 1782884,
                    "name": "Bob Powell"
                  },
                  "line_items": [
                    {
                      "id": 53334195,
                      "kind": "Service",
                      "description": "Phase 2 of the Online Store",
                      "quantity": 100,
                      "unit_price": 100,
                      "amount": 10000,
                      "taxed": true,
                      "taxed2": true
                    }
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update an estimate",
        "operationId": "updateEstimate",
        "description": "Updates the specific estimate by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns an estimate object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update an estimate",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimates/#update-an-estimate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update an estimate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Estimate"
                },
                "example": {
                  "id": 1439827,
                  "client_key": "ddd4504a68fb7339138d0c2ea89ba05a3cf12aa8",
                  "number": "1002",
                  "purchase_order": "2345",
                  "amount": 5000,
                  "tax": null,
                  "tax_amount": 0,
                  "tax2": null,
                  "tax2_amount": 0,
                  "discount": null,
                  "discount_amount": 0,
                  "subject": "Project Quote",
                  "notes": null,
                  "state": "draft",
                  "issue_date": null,
                  "sent_at": null,
                  "created_at": "2017-06-27T16:16:24Z",
                  "updated_at": "2017-06-27T16:17:06Z",
                  "accepted_at": null,
                  "declined_at": null,
                  "currency": "USD",
                  "client": {
                    "id": 5735774,
                    "name": "ABC Corp"
                  },
                  "creator": {
                    "id": 1782884,
                    "name": "Bob Powell"
                  },
                  "line_items": [
                    {
                      "id": 53339199,
                      "kind": "Service",
                      "description": "Project Description",
                      "quantity": 1,
                      "unit_price": 5000,
                      "amount": 5000,
                      "taxed": false,
                      "taxed2": false
                    }
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client this estimate belongs to.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "number": {
                    "type": "string",
                    "description": "If no value is set, the number will be automatically generated.",
                    "nullable": true
                  },
                  "purchase_order": {
                    "type": "string",
                    "description": "The purchase order number.",
                    "nullable": true
                  },
                  "tax": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "tax2": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "discount": {
                    "type": "number",
                    "description": "This percentage is subtracted from the subtotal. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "subject": {
                    "type": "string",
                    "description": "The estimate subject.",
                    "nullable": true
                  },
                  "notes": {
                    "type": "string",
                    "description": "Any additional notes to include on the estimate.",
                    "nullable": true
                  },
                  "currency": {
                    "type": "string",
                    "description": "The currency used by the estimate. If not provided, the client’s currency will be used. See a list of supported currencies",
                    "nullable": true
                  },
                  "issue_date": {
                    "type": "string",
                    "description": "Date the estimate was issued.",
                    "nullable": true,
                    "format": "date"
                  },
                  "line_items": {
                    "type": "array",
                    "description": "Array of line item parameters",
                    "nullable": true,
                    "items": {
                      "type": "object",
                      "properties": {
                        "id": {
                          "description": "Unique ID for the line item.",
                          "type": "integer",
                          "format": "int32"
                        },
                        "kind": {
                          "description": "The name of an estimate item category.",
                          "type": "string"
                        },
                        "description": {
                          "description": "Text description of the line item.",
                          "type": "string"
                        },
                        "quantity": {
                          "description": "The unit quantity of the item. Defaults to 1.",
                          "type": "integer",
                          "format": "int32"
                        },
                        "unit_price": {
                          "description": "The individual price per unit.",
                          "type": "number",
                          "format": "float"
                        },
                        "taxed": {
                          "description": "Whether the estimate’s tax percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        },
                        "taxed2": {
                          "description": "Whether the estimate’s tax2 percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/estimates/{estimateId}/messages": {
      "get": {
        "summary": "List all messages for an estimate",
        "operationId": "listMessagesForEstimate",
        "description": "Returns a list of messages associated with a given estimate. The estimate messages are returned sorted by creation date, with the most recently created messages appearing first.\n\nThe response contains an object with an estimate_messages property that contains an array of up to per_page messages. Each entry in the array is a separate message object. If no more messages are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your messages.",
        "externalDocs": {
          "description": "List all messages for an estimate",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-messages/#list-all-messages-for-an-estimate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all messages for an estimate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/EstimateMessages"
                },
                "example": {
                  "estimate_messages": [
                    {
                      "id": 2666236,
                      "sent_by": "Bob Powell",
                      "sent_by_email": "bobpowell@example.com",
                      "sent_from": "Bob Powell",
                      "sent_from_email": "bobpowell@example.com",
                      "send_me_a_copy": true,
                      "created_at": "2017-08-25T21:23:40Z",
                      "updated_at": "2017-08-25T21:23:40Z",
                      "recipients": [
                        {
                          "name": "Richard Roe",
                          "email": "richardroe@example.com"
                        },
                        {
                          "name": "Bob Powell",
                          "email": "bobpowell@example.com"
                        }
                      ],
                      "event_type": null,
                      "subject": "Estimate #1001 from API Examples",
                      "body": "---------------------------------------------\r\nEstimate Summary\r\n---------------------------------------------\r\nEstimate ID: 1001\r\nEstimate Date: 06/01/2017\r\nClient: 123 Industries\r\nP.O. Number: 5678\r\nAmount: $9,630.00\r\n\r\nYou can view the estimate here:\r\n\r\n%estimate_url%\r\n\r\nThank you!\r\n---------------------------------------------"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 1,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/estimates/1439818/messages?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/estimates/1439818/messages?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return estimate messages that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an estimate message or change estimate status",
        "operationId": "createEstimateMessage",
        "description": "Creates a new estimate message object. Returns an estimate message object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an estimate message",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-messages/#create-an-estimate-message"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an estimate message or change estimate status",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/EstimateMessage"
                },
                "example": {
                  "id": 2666240,
                  "sent_by": "Bob Powell",
                  "sent_by_email": "bobpowell@example.com",
                  "sent_from": "Bob Powell",
                  "sent_from_email": "bobpowell@example.com",
                  "send_me_a_copy": true,
                  "created_at": "2017-08-25T21:27:52Z",
                  "updated_at": "2017-08-25T21:27:52Z",
                  "recipients": [
                    {
                      "name": "Richard Roe",
                      "email": "richardroe@example.com"
                    },
                    {
                      "name": "Bob Powell",
                      "email": "bobpowell@example.com"
                    }
                  ],
                  "event_type": null,
                  "subject": "Estimate #1001",
                  "body": "Here is our estimate."
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "event_type": {
                    "type": "string",
                    "description": "If provided, runs an event against the estimate. Options: “accept”, “decline”, “re-open”, or “send”.",
                    "nullable": true
                  },
                  "recipients": {
                    "type": "array",
                    "description": "Array of recipient parameters. See below for details.",
                    "nullable": true,
                    "items": {
                      "type": "object",
                      "required": [
                        "email"
                      ],
                      "properties": {
                        "name": {
                          "description": "Name of the message recipient.",
                          "type": "string"
                        },
                        "email": {
                          "description": "Email of the message recipient.",
                          "type": "string",
                          "format": "email"
                        }
                      }
                    }
                  },
                  "subject": {
                    "type": "string",
                    "description": "The message subject.",
                    "nullable": true
                  },
                  "body": {
                    "type": "string",
                    "description": "The message body.",
                    "nullable": true
                  },
                  "send_me_a_copy": {
                    "type": "boolean",
                    "description": "If set to true, a copy of the message email will be sent to the current user. Defaults to false.",
                    "nullable": true
                  }
                },
                "required": [
                  "recipients"
                ]
              }
            }
          }
        }
      }
    },
    "/estimates/{estimateId}/messages/{messageId}": {
      "delete": {
        "summary": "Delete an estimate message",
        "operationId": "deleteEstimateMessage",
        "description": "Delete an estimate message. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an estimate message",
          "url": "https://help.getharvest.com/api-v2/estimates-api/estimates/estimate-messages/#delete-an-estimate-message"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an estimate message"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "estimateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "messageId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/expense_categories": {
      "get": {
        "summary": "List all expense categories",
        "operationId": "listExpenseCategories",
        "description": "Returns a list of your expense categories. The expense categories are returned sorted by creation date, with the most recently created expense categories appearing first.\n\nThe response contains an object with a expense_categories property that contains an array of up to per_page expense categories. Each entry in the array is a separate expense category object. If no more expense categories are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your expense categories.",
        "externalDocs": {
          "description": "List all expense categories",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expense-categories/#list-all-expense-categories"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all expense categories",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseCategories"
                },
                "example": {
                  "expense_categories": [
                    {
                      "id": 4197501,
                      "name": "Lodging",
                      "unit_name": null,
                      "unit_price": null,
                      "is_active": true,
                      "created_at": "2017-06-27T15:01:32Z",
                      "updated_at": "2017-06-27T15:01:32Z"
                    },
                    {
                      "id": 4195930,
                      "name": "Mileage",
                      "unit_name": "mile",
                      "unit_price": 0.535,
                      "is_active": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T20:41:00Z"
                    },
                    {
                      "id": 4195928,
                      "name": "Transportation",
                      "unit_name": null,
                      "unit_price": null,
                      "is_active": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T20:41:00Z"
                    },
                    {
                      "id": 4195926,
                      "name": "Meals",
                      "unit_name": null,
                      "unit_price": null,
                      "is_active": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T20:41:00Z"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 4,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/expense_categories?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/expense_categories?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "is_active",
            "description": "Pass true to only return active expense categories and false to return inactive expense categories.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return expense categories that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an expense category",
        "operationId": "createExpenseCategory",
        "description": "Creates a new expense category object. Returns an expense category object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an expense category",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expense-categories/#create-an-expense-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an expense category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseCategory"
                },
                "example": {
                  "id": 4197514,
                  "name": "Other",
                  "unit_name": null,
                  "unit_price": null,
                  "is_active": true,
                  "created_at": "2017-06-27T15:04:23Z",
                  "updated_at": "2017-06-27T15:04:23Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the expense category.",
                    "nullable": true
                  },
                  "unit_name": {
                    "type": "string",
                    "description": "The unit name of the expense category.",
                    "nullable": true
                  },
                  "unit_price": {
                    "type": "number",
                    "description": "The unit price of the expense category.",
                    "nullable": true,
                    "format": "float"
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the expense category is active or archived. Defaults to true.",
                    "nullable": true
                  }
                },
                "required": [
                  "name"
                ]
              }
            }
          }
        }
      }
    },
    "/expense_categories/{expenseCategoryId}": {
      "delete": {
        "summary": "Delete an expense category",
        "operationId": "deleteExpenseCategory",
        "description": "Delete an expense category. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an expense category",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expense-categories/#delete-an-expense-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an expense category"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "expenseCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve an expense category",
        "operationId": "retrieveExpenseCategory",
        "description": "Retrieves the expense category with the given ID. Returns an expense category object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve an expense category",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expense-categories/#retrieve-an-expense-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve an expense category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseCategory"
                },
                "example": {
                  "id": 4197501,
                  "name": "Lodging",
                  "unit_name": null,
                  "unit_price": null,
                  "is_active": true,
                  "created_at": "2017-06-27T15:01:32Z",
                  "updated_at": "2017-06-27T15:01:32Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "expenseCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update an expense category",
        "operationId": "updateExpenseCategory",
        "description": "Updates the specific expense category by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns an expense category object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update an expense category",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expense-categories/#update-an-expense-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update an expense category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseCategory"
                },
                "example": {
                  "id": 4197514,
                  "name": "Other",
                  "unit_name": null,
                  "unit_price": null,
                  "is_active": false,
                  "created_at": "2017-06-27T15:04:23Z",
                  "updated_at": "2017-06-27T15:04:58Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "expenseCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the expense category.",
                    "nullable": true
                  },
                  "unit_name": {
                    "type": "string",
                    "description": "The unit name of the expense category.",
                    "nullable": true
                  },
                  "unit_price": {
                    "type": "number",
                    "description": "The unit price of the expense category.",
                    "nullable": true,
                    "format": "float"
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the expense category is active or archived.",
                    "nullable": true
                  }
                }
              }
            }
          }
        }
      }
    },
    "/expenses": {
      "get": {
        "summary": "List all expenses",
        "operationId": "listExpenses",
        "description": "Returns a list of your expenses. If accessing this endpoint as an Administrator, all expenses in the account will be returned. If accessing this endpoint as a Manager, all expenses for assigned teammates and managed projects will be returned. The expenses are returned sorted by the spent_at date, with the most recent expenses appearing first.\n\nThe response contains an object with a expenses property that contains an array of up to per_page expenses. Each entry in the array is a separate expense object. If no more expenses are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your expenses.",
        "externalDocs": {
          "description": "List all expenses",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expenses/#list-all-expenses"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all expenses",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Expenses"
                },
                "example": {
                  "expenses": [
                    {
                      "id": 15296442,
                      "notes": "Lunch with client",
                      "total_cost": 33.35,
                      "units": 1,
                      "is_closed": false,
                      "is_locked": true,
                      "is_billed": true,
                      "locked_reason": "Expense is invoiced.",
                      "spent_date": "2017-03-03",
                      "created_at": "2017-06-27T15:09:54Z",
                      "updated_at": "2017-06-27T16:47:14Z",
                      "billable": true,
                      "receipt": {
                        "url": "https://{ACCOUNT_SUBDOMAIN}.harvestapp.com/expenses/15296442/receipt",
                        "file_name": "lunch_receipt.gif",
                        "file_size": 39410,
                        "content_type": "image/gif"
                      },
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      },
                      "user_assignment": {
                        "id": 125068553,
                        "is_project_manager": true,
                        "is_active": true,
                        "budget": null,
                        "created_at": "2017-06-26T22:32:52Z",
                        "updated_at": "2017-06-26T22:32:52Z",
                        "hourly_rate": 100
                      },
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "expense_category": {
                        "id": 4195926,
                        "name": "Meals",
                        "unit_price": null,
                        "unit_name": null
                      },
                      "client": {
                        "id": 5735774,
                        "name": "ABC Corp",
                        "currency": "USD"
                      },
                      "invoice": {
                        "id": 13150403,
                        "number": "1001"
                      }
                    },
                    {
                      "id": 15296423,
                      "notes": "Hotel stay for meeting",
                      "total_cost": 100,
                      "units": 1,
                      "is_closed": true,
                      "is_locked": true,
                      "is_billed": false,
                      "locked_reason": "The project is locked for this time period.",
                      "spent_date": "2017-03-01",
                      "created_at": "2017-06-27T15:09:17Z",
                      "updated_at": "2017-06-27T16:47:14Z",
                      "billable": true,
                      "receipt": null,
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      },
                      "user_assignment": {
                        "id": 125068554,
                        "is_project_manager": true,
                        "is_active": true,
                        "budget": null,
                        "created_at": "2017-06-26T22:32:52Z",
                        "updated_at": "2017-06-26T22:32:52Z",
                        "hourly_rate": 100
                      },
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "expense_category": {
                        "id": 4197501,
                        "name": "Lodging",
                        "unit_price": null,
                        "unit_name": null
                      },
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries",
                        "currency": "EUR"
                      },
                      "invoice": null
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/expenses?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/expenses?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "user_id",
            "description": "Only return expenses belonging to the user with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "client_id",
            "description": "Only return expenses belonging to the client with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "project_id",
            "description": "Only return expenses belonging to the project with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "is_billed",
            "description": "Pass true to only return expenses that have been invoiced and false to return expenses that have not been invoiced.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return expenses that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "from",
            "description": "Only return expenses with a spent_date on or after the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only return expenses with a spent_date on or before the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an expense",
        "operationId": "createExpense",
        "description": "Creates a new expense object. Returns an expense object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an expense",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expenses/#create-an-expense"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an expense",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Expense"
                },
                "example": {
                  "id": 15297032,
                  "notes": null,
                  "total_cost": 13.59,
                  "units": 1,
                  "is_closed": false,
                  "is_locked": false,
                  "is_billed": false,
                  "locked_reason": null,
                  "spent_date": "2017-03-01",
                  "created_at": "2017-06-27T15:42:27Z",
                  "updated_at": "2017-06-27T15:42:27Z",
                  "billable": true,
                  "receipt": null,
                  "user": {
                    "id": 1782959,
                    "name": "Kim Allen"
                  },
                  "user_assignment": {
                    "id": 125068553,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-06-26T22:32:52Z",
                    "updated_at": "2017-06-26T22:32:52Z",
                    "hourly_rate": 100
                  },
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "expense_category": {
                    "id": 4195926,
                    "name": "Meals",
                    "unit_price": null,
                    "unit_name": null
                  },
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries",
                    "currency": "EUR"
                  },
                  "invoice": null
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "user_id": {
                    "type": "integer",
                    "description": "The ID of the user associated with this expense. Defaults to the ID of the currently authenticated user.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "project_id": {
                    "type": "integer",
                    "description": "The ID of the project associated with this expense.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "expense_category_id": {
                    "type": "integer",
                    "description": "The ID of the expense category this expense is being tracked against.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "spent_date": {
                    "type": "string",
                    "description": "Date the expense occurred.",
                    "nullable": true,
                    "format": "date"
                  },
                  "units": {
                    "type": "integer",
                    "description": "The quantity of units to use in calculating the total_cost of the expense.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "total_cost": {
                    "type": "number",
                    "description": "The total amount of the expense.",
                    "nullable": true,
                    "format": "float"
                  },
                  "notes": {
                    "type": "string",
                    "description": "Textual notes used to describe the expense.",
                    "nullable": true
                  },
                  "billable": {
                    "type": "boolean",
                    "description": "Whether this expense is billable or not. Defaults to true.",
                    "nullable": true
                  },
                  "receipt": {
                    "type": "string",
                    "description": "A receipt file to attach to the expense. If including a receipt, you must submit a multipart/form-data request.",
                    "nullable": true
                  }
                },
                "required": [
                  "project_id",
                  "expense_category_id",
                  "spent_date"
                ]
              }
            }
          }
        }
      }
    },
    "/expenses/{expenseId}": {
      "delete": {
        "summary": "Delete an expense",
        "operationId": "deleteExpense",
        "description": "Delete an expense. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an expense",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expenses/#delete-an-expense"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an expense"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "expenseId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve an expense",
        "operationId": "retrieveExpense",
        "description": "Retrieves the expense with the given ID. Returns an expense object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve an expense",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expenses/#retrieve-an-expense"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve an expense",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Expense"
                },
                "example": {
                  "id": 15296442,
                  "notes": "Lunch with client",
                  "total_cost": 33.35,
                  "units": 1,
                  "is_closed": false,
                  "is_locked": true,
                  "is_billed": true,
                  "locked_reason": "Expense is invoiced.",
                  "spent_date": "2017-03-03",
                  "created_at": "2017-06-27T15:09:54Z",
                  "updated_at": "2017-06-27T16:47:14Z",
                  "billable": true,
                  "receipt": {
                    "url": "https://{ACCOUNT_SUBDOMAIN}.harvestapp.com/expenses/15296442/receipt",
                    "file_name": "lunch_receipt.gif",
                    "file_size": 39410,
                    "content_type": "image/gif"
                  },
                  "user": {
                    "id": 1782959,
                    "name": "Kim Allen"
                  },
                  "user_assignment": {
                    "id": 125068553,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-06-26T22:32:52Z",
                    "updated_at": "2017-06-26T22:32:52Z",
                    "hourly_rate": 100
                  },
                  "project": {
                    "id": 14307913,
                    "name": "Marketing Website",
                    "code": "MW"
                  },
                  "expense_category": {
                    "id": 4195926,
                    "name": "Meals",
                    "unit_price": null,
                    "unit_name": null
                  },
                  "client": {
                    "id": 5735774,
                    "name": "ABC Corp",
                    "currency": "USD"
                  },
                  "invoice": {
                    "id": 13150403,
                    "number": "1001"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "expenseId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update an expense",
        "operationId": "updateExpense",
        "description": "Updates the specific expense by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns an expense object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update an expense",
          "url": "https://help.getharvest.com/api-v2/expenses-api/expenses/expenses/#update-an-expense"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update an expense",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Expense"
                },
                "example": {
                  "id": 15297032,
                  "notes": "Dinner",
                  "total_cost": 13.59,
                  "units": 1,
                  "is_closed": false,
                  "is_locked": false,
                  "is_billed": false,
                  "locked_reason": null,
                  "spent_date": "2017-03-01",
                  "created_at": "2017-06-27T15:42:27Z",
                  "updated_at": "2017-06-27T15:45:51Z",
                  "billable": true,
                  "receipt": {
                    "url": "https://{ACCOUNT_SUBDOMAIN}.harvestapp.com/expenses/15297032/receipt",
                    "file_name": "dinner_receipt.gif",
                    "file_size": 39410,
                    "content_type": "image/gif"
                  },
                  "user": {
                    "id": 1782959,
                    "name": "Kim Allen"
                  },
                  "user_assignment": {
                    "id": 125068553,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-06-26T22:32:52Z",
                    "updated_at": "2017-06-26T22:32:52Z",
                    "hourly_rate": 100
                  },
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "expense_category": {
                    "id": 4195926,
                    "name": "Meals",
                    "unit_price": null,
                    "unit_name": null
                  },
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries",
                    "currency": "EUR"
                  },
                  "invoice": null
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "expenseId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "integer",
                    "description": "The ID of the project associated with this expense.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "expense_category_id": {
                    "type": "integer",
                    "description": "The ID of the expense category this expense is being tracked against.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "spent_date": {
                    "type": "string",
                    "description": "Date the expense occurred.",
                    "nullable": true,
                    "format": "date"
                  },
                  "units": {
                    "type": "integer",
                    "description": "The quantity of units to use in calculating the total_cost of the expense.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "total_cost": {
                    "type": "number",
                    "description": "The total amount of the expense.",
                    "nullable": true,
                    "format": "float"
                  },
                  "notes": {
                    "type": "string",
                    "description": "Textual notes used to describe the expense.",
                    "nullable": true
                  },
                  "billable": {
                    "type": "boolean",
                    "description": "Whether this expense is billable or not. Defaults to true.",
                    "nullable": true
                  },
                  "receipt": {
                    "type": "string",
                    "description": "A receipt file to attach to the expense. If including a receipt, you must submit a multipart/form-data request.",
                    "nullable": true
                  },
                  "delete_receipt": {
                    "type": "boolean",
                    "description": "Whether an attached expense receipt should be deleted. Pass true to delete the expense receipt.",
                    "nullable": true
                  }
                }
              }
            }
          }
        }
      }
    },
    "/invoice_item_categories": {
      "get": {
        "summary": "List all invoice item categories",
        "operationId": "listInvoiceItemCategories",
        "description": "Returns a list of your invoice item categories. The invoice item categories are returned sorted by creation date, with the most recently created invoice item categories appearing first.\n\nThe response contains an object with a invoice_item_categories property that contains an array of up to per_page invoice item categories. Each entry in the array is a separate invoice item category object. If no more invoice item categories are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your invoice item categories.",
        "externalDocs": {
          "description": "List all invoice item categories",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-item-categories/#list-all-invoice-item-categories"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all invoice item categories",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoiceItemCategories"
                },
                "example": {
                  "invoice_item_categories": [
                    {
                      "id": 1466293,
                      "name": "Product",
                      "use_as_service": false,
                      "use_as_expense": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T20:41:00Z"
                    },
                    {
                      "id": 1466292,
                      "name": "Service",
                      "use_as_service": true,
                      "use_as_expense": false,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T20:41:00Z"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/invoice_item_categories?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/invoice_item_categories?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "updated_since",
            "description": "Only return invoice item categories that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an invoice item category",
        "operationId": "createInvoiceItemCategory",
        "description": "Creates a new invoice item category object. Returns an invoice item category object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an invoice item category",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-item-categories/#create-an-invoice-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an invoice item category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoiceItemCategory"
                },
                "example": {
                  "id": 1467098,
                  "name": "Hosting",
                  "use_as_service": false,
                  "use_as_expense": false,
                  "created_at": "2017-06-27T16:20:59Z",
                  "updated_at": "2017-06-27T16:20:59Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the invoice item category.",
                    "nullable": true
                  }
                },
                "required": [
                  "name"
                ]
              }
            }
          }
        }
      }
    },
    "/invoice_item_categories/{invoiceItemCategoryId}": {
      "delete": {
        "summary": "Delete an invoice item category",
        "operationId": "deleteInvoiceItemCategory",
        "description": "Delete an invoice item category. Deleting an invoice item category is only possible if use_as_service and use_as_expense are both false. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an invoice item category",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-item-categories/#delete-an-invoice-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an invoice item category"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceItemCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve an invoice item category",
        "operationId": "retrieveInvoiceItemCategory",
        "description": "Retrieves the invoice item category with the given ID. Returns an invoice item category object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve an invoice item category",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-item-categories/#retrieve-an-invoice-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve an invoice item category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoiceItemCategory"
                },
                "example": {
                  "id": 1466293,
                  "name": "Product",
                  "use_as_service": false,
                  "use_as_expense": true,
                  "created_at": "2017-06-26T20:41:00Z",
                  "updated_at": "2017-06-26T20:41:00Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceItemCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update an invoice item category",
        "operationId": "updateInvoiceItemCategory",
        "description": "Updates the specific invoice item category by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns an invoice item category object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update an invoice item category",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-item-categories/#update-an-invoice-item-category"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update an invoice item category",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoiceItemCategory"
                },
                "example": {
                  "id": 1467098,
                  "name": "Expense",
                  "use_as_service": false,
                  "use_as_expense": false,
                  "created_at": "2017-06-27T16:20:59Z",
                  "updated_at": "2017-06-27T16:21:26Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceItemCategoryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the invoice item category.",
                    "nullable": true
                  }
                }
              }
            }
          }
        }
      }
    },
    "/invoices": {
      "get": {
        "summary": "List all invoices",
        "operationId": "listInvoices",
        "description": "Returns a list of your invoices. The invoices are returned sorted by issue date, with the most recently issued invoices appearing first.\n\nThe response contains an object with a invoices property that contains an array of up to per_page invoices. Each entry in the array is a separate invoice object. If no more invoices are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your invoices.",
        "externalDocs": {
          "description": "List all invoices",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoices/#list-all-invoices"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all invoices",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Invoices"
                },
                "example": {
                  "invoices": [
                    {
                      "id": 13150403,
                      "client_key": "21312da13d457947a217da6775477afee8c2eba8",
                      "number": "1001",
                      "purchase_order": "",
                      "amount": 288.9,
                      "due_amount": 288.9,
                      "tax": 5,
                      "tax_amount": 13.5,
                      "tax2": 2,
                      "tax2_amount": 5.4,
                      "discount": 10,
                      "discount_amount": 30,
                      "subject": "Online Store - Phase 1",
                      "notes": "Some notes about the invoice.",
                      "state": "open",
                      "period_start": "2017-03-01",
                      "period_end": "2017-03-01",
                      "issue_date": "2017-04-01",
                      "due_date": "2017-04-01",
                      "payment_term": "upon receipt",
                      "sent_at": "2017-08-23T22:25:59Z",
                      "paid_at": null,
                      "paid_date": null,
                      "closed_at": null,
                      "recurring_invoice_id": null,
                      "created_at": "2017-06-27T16:27:16Z",
                      "updated_at": "2017-08-23T22:25:59Z",
                      "currency": "EUR",
                      "payment_options": [
                        "credit_card"
                      ],
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "estimate": null,
                      "retainer": null,
                      "creator": {
                        "id": 1782884,
                        "name": "Bob Powell"
                      },
                      "line_items": [
                        {
                          "id": 53341602,
                          "kind": "Service",
                          "description": "03/01/2017 - Project Management: [9:00am - 11:00am] Planning meetings",
                          "quantity": 2,
                          "unit_price": 100,
                          "amount": 200,
                          "taxed": true,
                          "taxed2": true,
                          "project": {
                            "id": 14308069,
                            "name": "Online Store - Phase 1",
                            "code": "OS1"
                          }
                        },
                        {
                          "id": 53341603,
                          "kind": "Service",
                          "description": "03/01/2017 - Programming: [1:00pm - 2:00pm] Importing products",
                          "quantity": 1,
                          "unit_price": 100,
                          "amount": 100,
                          "taxed": true,
                          "taxed2": true,
                          "project": {
                            "id": 14308069,
                            "name": "Online Store - Phase 1",
                            "code": "OS1"
                          }
                        }
                      ]
                    },
                    {
                      "id": 13150378,
                      "client_key": "9e97f4a65c5b83b1fc02f54e5a41c9dc7d458542",
                      "number": "1000",
                      "purchase_order": "1234",
                      "amount": 10700,
                      "due_amount": 0,
                      "tax": 5,
                      "tax_amount": 500,
                      "tax2": 2,
                      "tax2_amount": 200,
                      "discount": null,
                      "discount_amount": 0,
                      "subject": "Online Store - Phase 1",
                      "notes": "Some notes about the invoice.",
                      "state": "paid",
                      "period_start": null,
                      "period_end": null,
                      "issue_date": "2017-02-01",
                      "due_date": "2017-03-03",
                      "payment_term": "custom",
                      "sent_at": "2017-02-01T07:00:00Z",
                      "paid_at": "2017-02-21T00:00:00Z",
                      "paid_date": "2017-02-21",
                      "closed_at": null,
                      "recurring_invoice_id": null,
                      "created_at": "2017-06-27T16:24:30Z",
                      "updated_at": "2017-06-27T16:24:57Z",
                      "currency": "USD",
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "estimate": {
                        "id": 1439814
                      },
                      "retainer": null,
                      "creator": {
                        "id": 1782884,
                        "name": "Bob Powell"
                      },
                      "line_items": [
                        {
                          "id": 53341450,
                          "kind": "Service",
                          "description": "50% of Phase 1 of the Online Store",
                          "quantity": 100,
                          "unit_price": 100,
                          "amount": 10000,
                          "taxed": true,
                          "taxed2": true,
                          "project": {
                            "id": 14308069,
                            "name": "Online Store - Phase 1",
                            "code": "OS1"
                          }
                        }
                      ]
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/invoices?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/invoices?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "client_id",
            "description": "Only return invoices belonging to the client with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "project_id",
            "description": "Only return invoices associated with the project with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return invoices that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "from",
            "description": "Only return invoices with an issue_date on or after the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only return invoices with an issue_date on or before the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "state",
            "description": "Only return invoices with a state matching the value provided. Options: draft, open, paid, or closed.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an invoice",
        "operationId": "createInvoice",
        "description": "Creates a new invoice object. Returns an invoice object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a free-form invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoices/#create-a-free-form-invoice"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an invoice",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Invoice"
                },
                "example": {
                  "id": 13150453,
                  "client_key": "8b86437630b6c260c1bfa289f0154960f83b606d",
                  "number": "1002",
                  "purchase_order": null,
                  "amount": 5000,
                  "due_amount": 5000,
                  "tax": null,
                  "tax_amount": 0,
                  "tax2": null,
                  "tax2_amount": 0,
                  "discount": null,
                  "discount_amount": 0,
                  "subject": "ABC Project Quote",
                  "notes": null,
                  "state": "draft",
                  "period_start": null,
                  "period_end": null,
                  "issue_date": "2017-06-27",
                  "due_date": "2017-07-27",
                  "payment_term": "custom",
                  "sent_at": null,
                  "paid_at": null,
                  "paid_date": null,
                  "closed_at": null,
                  "recurring_invoice_id": null,
                  "created_at": "2017-06-27T16:34:24Z",
                  "updated_at": "2017-06-27T16:34:24Z",
                  "currency": "USD",
                  "payment_options": [
                    "credit_card"
                  ],
                  "client": {
                    "id": 5735774,
                    "name": "ABC Corp"
                  },
                  "estimate": null,
                  "retainer": null,
                  "creator": {
                    "id": 1782884,
                    "name": "Bob Powell"
                  },
                  "line_items": [
                    {
                      "id": 53341928,
                      "kind": "Service",
                      "description": "ABC Project",
                      "quantity": 1,
                      "unit_price": 5000,
                      "amount": 5000,
                      "taxed": false,
                      "taxed2": false,
                      "project": null
                    }
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client this invoice belongs to.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "estimate_id": {
                    "type": "integer",
                    "description": "The ID of the estimate associated with this invoice.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "number": {
                    "type": "string",
                    "description": "If no value is set, the number will be automatically generated.",
                    "nullable": true
                  },
                  "purchase_order": {
                    "type": "string",
                    "description": "The purchase order number.",
                    "nullable": true
                  },
                  "tax": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "tax2": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "discount": {
                    "type": "number",
                    "description": "This percentage is subtracted from the subtotal. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "subject": {
                    "type": "string",
                    "description": "The invoice subject.",
                    "nullable": true
                  },
                  "notes": {
                    "type": "string",
                    "description": "Any additional notes to include on the invoice.",
                    "nullable": true
                  },
                  "currency": {
                    "type": "string",
                    "description": "The currency used by the invoice. If not provided, the client’s currency will be used. See a list of supported currencies",
                    "nullable": true
                  },
                  "issue_date": {
                    "type": "string",
                    "description": "Date the invoice was issued. Defaults to today’s date.",
                    "nullable": true,
                    "format": "date"
                  },
                  "due_date": {
                    "type": "string",
                    "description": "Date the invoice is due. Defaults to the issue_date if no payment_term is specified. To set a custom due_date the payment_term must also be set to custom, otherwise the value supplied in the request for due_date will be ignored and the due_date will be calculated using the issue_date and the payment_term.",
                    "nullable": true,
                    "format": "date"
                  },
                  "payment_term": {
                    "type": "string",
                    "description": "The timeframe in which the invoice should be paid. Defaults to custom. Options: upon receipt, net 15, net 30, net 45, net 60, or custom.",
                    "nullable": true
                  },
                  "payment_options": {
                    "type": "array",
                    "description": "The payment options available to pay the invoice. Your account must be configured with the appropriate options under Settings > Integrations > Online payment to assign them. Options: [ach, credit_card, paypal]",
                    "nullable": true,
                    "items": {
                      "type": "string",
                      "enum": [
                        "ach",
                        "credit_card",
                        "paypal"
                      ]
                    }
                  },
                  "line_items_import": {
                    "type": "object",
                    "description": "An line items import object",
                    "nullable": true,
                    "required": [
                      "project_ids"
                    ],
                    "properties": {
                      "project_ids": {
                        "description": "An array of the client’s project IDs you’d like to include time/expenses from.",
                        "type": "array",
                        "items": {
                          "type": "integer"
                        }
                      },
                      "time": {
                        "description": "An time import object.",
                        "type": "object",
                        "required": [
                          "summary_type"
                        ],
                        "properties": {
                          "summary_type": {
                            "type": "string",
                            "description": "How to summarize the time entries per line item. Options: project, task, people, or detailed."
                          },
                          "from": {
                            "type": "string",
                            "format": "date",
                            "description": "Start date for included time entries. Must be provided if to is present. If neither from or to are provided, all unbilled time entries will be included."
                          },
                          "to": {
                            "type": "string",
                            "format": "date",
                            "description": "End date for included time entries. Must be provided if from is present. If neither from or to are provided, all unbilled time entries will be included."
                          }
                        }
                      },
                      "expenses": {
                        "description": "An expense import object.",
                        "type": "object",
                        "required": [
                          "summary_type"
                        ],
                        "properties": {
                          "summary_type": {
                            "type": "string",
                            "description": "How to summarize the expenses per line item. Options: project, category, people, or detailed."
                          },
                          "from": {
                            "type": "string",
                            "format": "date",
                            "description": "Start date for included expenses. Must be provided if to is present. If neither from or to are provided, all unbilled expenses will be included."
                          },
                          "to": {
                            "type": "string",
                            "format": "date",
                            "description": "End date for included expenses. Must be provided if from is present. If neither from or to are provided, all unbilled expenses will be included."
                          },
                          "attach_receipt": {
                            "type": "boolean",
                            "description": "If set to true, a PDF containing an expense report with receipts will be attached to the invoice. Defaults to false."
                          }
                        }
                      }
                    }
                  },
                  "retainer_id": {
                    "type": "integer",
                    "description": "The ID of the retainer you want to add funds to with this invoice. Note: retainers cannot be fully used (created, drawn against, closed, etc.) via the API at this time. The only available action is to add funds.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "line_items": {
                    "type": "array",
                    "description": "Array of line item parameters",
                    "nullable": true,
                    "items": {
                      "type": "object",
                      "required": [
                        "kind",
                        "unit_price"
                      ],
                      "properties": {
                        "project_id": {
                          "description": "The ID of the project associated with this line item.",
                          "type": "integer",
                          "format": "int32"
                        },
                        "kind": {
                          "description": "The name of an invoice item category.",
                          "type": "string"
                        },
                        "description": {
                          "description": "Text description of the line item.",
                          "type": "string"
                        },
                        "quantity": {
                          "description": "The unit quantity of the item. Defaults to 1.",
                          "type": "number",
                          "format": "float"
                        },
                        "unit_price": {
                          "description": "The individual price per unit.",
                          "type": "number",
                          "format": "float"
                        },
                        "taxed": {
                          "description": "Whether the invoice’s tax percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        },
                        "taxed2": {
                          "description": "Whether the invoice’s tax2 percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        }
                      }
                    }
                  }
                },
                "required": [
                  "client_id"
                ]
              }
            }
          }
        }
      }
    },
    "/invoices/{invoiceId}": {
      "delete": {
        "summary": "Delete an invoice",
        "operationId": "deleteInvoice",
        "description": "Delete an invoice. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoices/#delete-an-invoice"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an invoice"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve an invoice",
        "operationId": "retrieveInvoice",
        "description": "Retrieves the invoice with the given ID. Returns an invoice object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve an invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoices/#retrieve-an-invoice"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve an invoice",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Invoice"
                },
                "example": {
                  "id": 13150378,
                  "client_key": "9e97f4a65c5b83b1fc02f54e5a41c9dc7d458542",
                  "number": "1000",
                  "purchase_order": "1234",
                  "amount": 10700,
                  "due_amount": 0,
                  "tax": 5,
                  "tax_amount": 500,
                  "tax2": 2,
                  "tax2_amount": 200,
                  "discount": null,
                  "discount_amount": 0,
                  "subject": "Online Store - Phase 1",
                  "notes": "Some notes about the invoice.",
                  "state": "paid",
                  "period_start": null,
                  "period_end": null,
                  "issue_date": "2017-02-01",
                  "due_date": "2017-03-03",
                  "payment_term": "custom",
                  "sent_at": "2017-02-01T07:00:00Z",
                  "paid_at": "2017-02-21T00:00:00Z",
                  "paid_date": "2017-02-21",
                  "closed_at": null,
                  "recurring_invoice_id": null,
                  "created_at": "2017-06-27T16:24:30Z",
                  "updated_at": "2017-06-27T16:24:57Z",
                  "currency": "USD",
                  "payment_options": [
                    "credit_card"
                  ],
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries"
                  },
                  "estimate": {
                    "id": 1439814
                  },
                  "retainer": null,
                  "creator": {
                    "id": 1782884,
                    "name": "Bob Powell"
                  },
                  "line_items": [
                    {
                      "id": 53341450,
                      "kind": "Service",
                      "description": "50% of Phase 1 of the Online Store",
                      "quantity": 100,
                      "unit_price": 100,
                      "amount": 10000,
                      "taxed": true,
                      "taxed2": true,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      }
                    }
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update an invoice",
        "operationId": "updateInvoice",
        "description": "Updates the specific invoice by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns an invoice object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update an invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoices/#update-an-invoice"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update an invoice",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Invoice"
                },
                "example": {
                  "id": 13150453,
                  "client_key": "8b86437630b6c260c1bfa289f0154960f83b606d",
                  "number": "1002",
                  "purchase_order": "2345",
                  "amount": 5000,
                  "due_amount": 5000,
                  "tax": null,
                  "tax_amount": 0,
                  "tax2": null,
                  "tax2_amount": 0,
                  "discount": null,
                  "discount_amount": 0,
                  "subject": "ABC Project Quote",
                  "notes": null,
                  "state": "draft",
                  "period_start": null,
                  "period_end": null,
                  "issue_date": "2017-06-27",
                  "due_date": "2017-07-27",
                  "payment_term": "custom",
                  "sent_at": null,
                  "paid_at": null,
                  "paid_date": null,
                  "closed_at": null,
                  "recurring_invoice_id": null,
                  "created_at": "2017-06-27T16:34:24Z",
                  "updated_at": "2017-06-27T16:36:33Z",
                  "currency": "USD",
                  "payment_options": [
                    "credit_card"
                  ],
                  "client": {
                    "id": 5735774,
                    "name": "ABC Corp"
                  },
                  "estimate": null,
                  "retainer": null,
                  "creator": {
                    "id": 1782884,
                    "name": "Bob Powell"
                  },
                  "line_items": [
                    {
                      "id": 53341928,
                      "kind": "Service",
                      "description": "ABC Project",
                      "quantity": 1,
                      "unit_price": 5000,
                      "amount": 5000,
                      "taxed": false,
                      "taxed2": false,
                      "project": null
                    }
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client this invoice belongs to.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "retainer_id": {
                    "type": "integer",
                    "description": "The ID of the retainer associated with this invoice.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "estimate_id": {
                    "type": "integer",
                    "description": "The ID of the estimate associated with this invoice.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "number": {
                    "type": "string",
                    "description": "If no value is set, the number will be automatically generated.",
                    "nullable": true
                  },
                  "purchase_order": {
                    "type": "string",
                    "description": "The purchase order number.",
                    "nullable": true
                  },
                  "tax": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "tax2": {
                    "type": "number",
                    "description": "This percentage is applied to the subtotal, including line items and discounts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "discount": {
                    "type": "number",
                    "description": "This percentage is subtracted from the subtotal. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "subject": {
                    "type": "string",
                    "description": "The invoice subject.",
                    "nullable": true
                  },
                  "notes": {
                    "type": "string",
                    "description": "Any additional notes to include on the invoice.",
                    "nullable": true
                  },
                  "currency": {
                    "type": "string",
                    "description": "The currency used by the invoice. If not provided, the client’s currency will be used. See a list of supported currencies",
                    "nullable": true
                  },
                  "issue_date": {
                    "type": "string",
                    "description": "Date the invoice was issued.",
                    "nullable": true,
                    "format": "date"
                  },
                  "due_date": {
                    "type": "string",
                    "description": "Date the invoice is due.",
                    "nullable": true,
                    "format": "date"
                  },
                  "payment_term": {
                    "type": "string",
                    "description": "The timeframe in which the invoice should be paid. Options: upon receipt, net 15, net 30, net 45, or net 60.",
                    "nullable": true
                  },
                  "payment_options": {
                    "type": "array",
                    "description": "The payment options available to pay the invoice. Your account must be configured with the appropriate options under Settings > Integrations > Online payment to assign them. Options: [ach, credit_card, paypal]",
                    "nullable": true,
                    "items": {
                      "type": "string",
                      "enum": [
                        "ach",
                        "credit_card",
                        "paypal"
                      ]
                    }
                  },
                  "line_items": {
                    "type": "array",
                    "description": "Array of line item parameters",
                    "nullable": true,
                    "items": {
                      "type": "object",
                      "properties": {
                        "id": {
                          "description": "Unique ID for the line item.",
                          "type": "integer",
                          "format": "int32"
                        },
                        "project_id": {
                          "description": "The ID of the project associated with this line item.",
                          "type": "integer",
                          "format": "int32"
                        },
                        "kind": {
                          "description": "The name of an invoice item category.",
                          "type": "string"
                        },
                        "description": {
                          "description": "Text description of the line item.",
                          "type": "string"
                        },
                        "quantity": {
                          "description": "The unit quantity of the item. Defaults to 1.",
                          "type": "number",
                          "format": "float"
                        },
                        "unit_price": {
                          "description": "The individual price per unit.",
                          "type": "number",
                          "format": "float"
                        },
                        "taxed": {
                          "description": "Whether the invoice’s tax percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        },
                        "taxed2": {
                          "description": "Whether the invoice’s tax2 percentage applies to this line item. Defaults to false.",
                          "type": "boolean"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/invoices/{invoiceId}/messages": {
      "get": {
        "summary": "List all messages for an invoice",
        "operationId": "listMessagesForInvoice",
        "description": "Returns a list of messages associated with a given invoice. The invoice messages are returned sorted by creation date, with the most recently created messages appearing first.\n\nThe response contains an object with an invoice_messages property that contains an array of up to per_page messages. Each entry in the array is a separate message object. If no more messages are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your messages.",
        "externalDocs": {
          "description": "List all messages for an invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-messages/#list-all-messages-for-an-invoice"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all messages for an invoice",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoiceMessages"
                },
                "example": {
                  "invoice_messages": [
                    {
                      "id": 27835209,
                      "sent_by": "Bob Powell",
                      "sent_by_email": "bobpowell@example.com",
                      "sent_from": "Bob Powell",
                      "sent_from_email": "bobpowell@example.com",
                      "include_link_to_client_invoice": false,
                      "send_me_a_copy": false,
                      "thank_you": false,
                      "reminder": false,
                      "send_reminder_on": null,
                      "created_at": "2017-08-23T22:15:06Z",
                      "updated_at": "2017-08-23T22:15:06Z",
                      "attach_pdf": true,
                      "event_type": null,
                      "recipients": [
                        {
                          "name": "Richard Roe",
                          "email": "richardroe@example.com"
                        }
                      ],
                      "subject": "Past due invoice reminder: #1001 from API Examples",
                      "body": "Dear Customer,\r\n\r\nThis is a friendly reminder to let you know that Invoice 1001 is 144 days past due. If you have already sent the payment, please disregard this message. If not, we would appreciate your prompt attention to this matter.\r\n\r\nThank you for your business.\r\n\r\nCheers,\r\nAPI Examples"
                    },
                    {
                      "id": 27835207,
                      "sent_by": "Bob Powell",
                      "sent_by_email": "bobpowell@example.com",
                      "sent_from": "Bob Powell",
                      "sent_from_email": "bobpowell@example.com",
                      "include_link_to_client_invoice": false,
                      "send_me_a_copy": true,
                      "thank_you": false,
                      "reminder": false,
                      "send_reminder_on": null,
                      "created_at": "2017-08-23T22:14:49Z",
                      "updated_at": "2017-08-23T22:14:49Z",
                      "attach_pdf": true,
                      "event_type": null,
                      "recipients": [
                        {
                          "name": "Richard Roe",
                          "email": "richardroe@example.com"
                        },
                        {
                          "name": "Bob Powell",
                          "email": "bobpowell@example.com"
                        }
                      ],
                      "subject": "Invoice #1001 from API Examples",
                      "body": "---------------------------------------------\r\nInvoice Summary\r\n---------------------------------------------\r\nInvoice ID: 1001\r\nIssue Date: 04/01/2017\r\nClient: 123 Industries\r\nP.O. Number: \r\nAmount: €288.90\r\nDue: 04/01/2017 (upon receipt)\r\n\r\nThe detailed invoice is attached as a PDF.\r\n\r\nThank you!\r\n---------------------------------------------"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/api/v2/invoices/13150403/messages?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/invoices/13150403/messages?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return invoice messages that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an invoice message or change invoice status",
        "operationId": "createInvoiceMessage",
        "description": "Creates a new invoice message object. Returns an invoice message object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an invoice message",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-messages/#create-an-invoice-message"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an invoice message or change invoice status",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoiceMessage"
                },
                "example": {
                  "id": 27835324,
                  "sent_by": "Bob Powell",
                  "sent_by_email": "bobpowell@example.com",
                  "sent_from": "Bob Powell",
                  "sent_from_email": "bobpowell@example.com",
                  "include_link_to_client_invoice": false,
                  "send_me_a_copy": true,
                  "thank_you": false,
                  "reminder": false,
                  "send_reminder_on": null,
                  "created_at": "2017-08-23T22:25:59Z",
                  "updated_at": "2017-08-23T22:25:59Z",
                  "attach_pdf": true,
                  "event_type": null,
                  "recipients": [
                    {
                      "name": "Richard Roe",
                      "email": "richardroe@example.com"
                    },
                    {
                      "name": "Bob Powell",
                      "email": "bobpowell@example.com"
                    }
                  ],
                  "subject": "Invoice #1001",
                  "body": "The invoice is attached below."
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "event_type": {
                    "type": "string",
                    "description": "If provided, runs an event against the invoice. Options: close, draft, re-open, or send.",
                    "nullable": true
                  },
                  "recipients": {
                    "type": "array",
                    "description": "Array of recipient parameters. See below for details.",
                    "nullable": true,
                    "items": {
                      "type": "object",
                      "required": [
                        "email"
                      ],
                      "properties": {
                        "name": {
                          "description": "Name of the message recipient.",
                          "type": "string"
                        },
                        "email": {
                          "description": "Email of the message recipient.",
                          "type": "string",
                          "format": "email"
                        }
                      }
                    }
                  },
                  "subject": {
                    "type": "string",
                    "description": "The message subject.",
                    "nullable": true
                  },
                  "body": {
                    "type": "string",
                    "description": "The message body.",
                    "nullable": true
                  },
                  "include_link_to_client_invoice": {
                    "type": "boolean",
                    "description": "DEPRECATED A link to the client invoice URL will be automatically included in the message email if payment_options have been assigned to the invoice. Setting to true will be ignored. Setting to false will clear all payment_options on the invoice.",
                    "nullable": true,
                    "deprecated": true
                  },
                  "attach_pdf": {
                    "type": "boolean",
                    "description": "If set to true, a PDF of the invoice will be attached to the message email. Defaults to false.",
                    "nullable": true
                  },
                  "send_me_a_copy": {
                    "type": "boolean",
                    "description": "If set to true, a copy of the message email will be sent to the current user. Defaults to false.",
                    "nullable": true
                  },
                  "thank_you": {
                    "type": "boolean",
                    "description": "If set to true, a thank you message email will be sent. Defaults to false.",
                    "nullable": true
                  }
                },
                "required": [
                  "recipients"
                ]
              }
            }
          }
        }
      }
    },
    "/invoices/{invoiceId}/messages/new": {
      "get": {
        "summary": "Retrieve invoice message subject and body for specific invoice",
        "operationId": "retrieveInvoiceMessageSubjectAndBodyForSpecificInvoice",
        "description": "Returns the subject and body text as configured in Harvest of an invoice message for a specific invoice and a 200 OK response code if the call succeeded. Does not create the invoice message. If no parameters are passed, will return the subject and body of a general invoice message for the specific invoice.",
        "externalDocs": {
          "description": "Retrieve invoice message subject and body for specific invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-messages/#retrieve-invoice-message-subject-and-body-for-specific-invoice"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve invoice message subject and body for specific invoice",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoiceMessageSubjectAndBody"
                },
                "example": {
                  "invoice_id": 13150403,
                  "subject": "Past due invoice reminder: #1002 from API Examples",
                  "body": "Dear Customer,\n\nThis is a friendly reminder to let you know that Invoice 1002 is 20 days past due. If you have already sent the payment, please disregard this message. If not, we would appreciate your prompt attention to this matter.\n\nThank you for your business.\n\nCheers,\nAPI Examples\n",
                  "reminder": false,
                  "thank_you": false
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "thank_you",
            "description": "Set to true to return the subject and body of a thank-you invoice message for the specific invoice.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "reminder",
            "description": "Set to true to return the subject and body of a reminder invoice message for the specific invoice.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          }
        ]
      }
    },
    "/invoices/{invoiceId}/messages/{messageId}": {
      "delete": {
        "summary": "Delete an invoice message",
        "operationId": "deleteInvoiceMessage",
        "description": "Delete an invoice message. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an invoice message",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-messages/#delete-an-invoice-message"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an invoice message"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "messageId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/invoices/{invoiceId}/payments": {
      "get": {
        "summary": "List all payments for an invoice",
        "operationId": "listPaymentsForInvoice",
        "description": "Returns a list of payments associate with a given invoice. The payments are returned sorted by creation date, with the most recently created payments appearing first.\n\nThe response contains an object with an invoice_payments property that contains an array of up to per_page payments. Each entry in the array is a separate payment object. If no more payments are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your payments.",
        "externalDocs": {
          "description": "List all payments for an invoice",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-payments/#list-all-payments-for-an-invoice"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all payments for an invoice",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoicePayments"
                },
                "example": {
                  "invoice_payments": [
                    {
                      "id": 10112854,
                      "amount": 10700,
                      "paid_at": "2017-02-21T00:00:00Z",
                      "paid_date": "2017-02-21",
                      "recorded_by": "Alice Doe",
                      "recorded_by_email": "alice@example.com",
                      "notes": "Paid via check #4321",
                      "transaction_id": null,
                      "created_at": "2017-06-27T16:24:57Z",
                      "updated_at": "2017-06-27T16:24:57Z",
                      "payment_gateway": {
                        "id": 1234,
                        "name": "Linkpoint International"
                      }
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 1,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/invoices/13150378/payments?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/invoices/13150378/payments?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return invoice payments that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create an invoice payment",
        "operationId": "createInvoicePayment",
        "description": "Creates a new invoice payment object. Returns an invoice payment object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create an invoice payment",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-payments/#create-an-invoice-payment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create an invoice payment",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InvoicePayment"
                },
                "example": {
                  "id": 10336386,
                  "amount": 1575.86,
                  "paid_at": "2017-07-24T13:32:18Z",
                  "paid_date": "2017-07-24",
                  "recorded_by": "Jane Bar",
                  "recorded_by_email": "jane@example.com",
                  "notes": "Paid by phone",
                  "transaction_id": null,
                  "created_at": "2017-07-28T14:42:44Z",
                  "updated_at": "2017-07-28T14:42:44Z",
                  "payment_gateway": {
                    "id": null,
                    "name": null
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "amount": {
                    "type": "number",
                    "description": "The amount of the payment.",
                    "nullable": true,
                    "format": "float"
                  },
                  "paid_at": {
                    "type": "string",
                    "description": "Date and time the payment was made. Pass either paid_at or paid_date, but not both.",
                    "nullable": true,
                    "format": "date-time"
                  },
                  "paid_date": {
                    "type": "string",
                    "description": "Date the payment was made. Pass either paid_at or paid_date, but not both.",
                    "nullable": true,
                    "format": "date"
                  },
                  "notes": {
                    "type": "string",
                    "description": "Any notes to be associated with the payment.",
                    "nullable": true
                  },
                  "send_thank_you": {
                    "type": "boolean",
                    "description": "Whether or not to send a thank you email (if enabled for your account in Invoices > Configure > Messages). Only sends an email if the invoice will be fully paid after creating this payment. Defaults to true.",
                    "nullable": true
                  }
                },
                "required": [
                  "amount"
                ]
              }
            }
          }
        }
      }
    },
    "/invoices/{invoiceId}/payments/{paymentId}": {
      "delete": {
        "summary": "Delete an invoice payment",
        "operationId": "deleteInvoicePayment",
        "description": "Delete an invoice payment. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete an invoice payment",
          "url": "https://help.getharvest.com/api-v2/invoices-api/invoices/invoice-payments/#delete-an-invoice-payment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete an invoice payment"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "invoiceId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "paymentId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/projects": {
      "get": {
        "summary": "List all projects",
        "operationId": "listProjects",
        "description": "Returns a list of your projects. The projects are returned sorted by creation date, with the most recently created projects appearing first.\n\nThe response contains an object with a projects property that contains an array of up to per_page projects. Each entry in the array is a separate project object. If no more projects are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your projects.",
        "externalDocs": {
          "description": "List all projects",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/projects/#list-all-projects"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all projects",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Projects"
                },
                "example": {
                  "projects": [
                    {
                      "id": 14308069,
                      "name": "Online Store - Phase 1",
                      "code": "OS1",
                      "is_active": true,
                      "bill_by": "Project",
                      "budget": 200,
                      "budget_by": "project",
                      "budget_is_monthly": false,
                      "notify_when_over_budget": true,
                      "over_budget_notification_percentage": 80,
                      "over_budget_notification_date": null,
                      "show_budget_to_all": false,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:54:06Z",
                      "starts_on": "2017-06-01",
                      "ends_on": null,
                      "is_billable": true,
                      "is_fixed_fee": false,
                      "notes": "",
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries",
                        "currency": "EUR"
                      },
                      "cost_budget": null,
                      "cost_budget_include_expenses": false,
                      "hourly_rate": 100,
                      "fee": null
                    },
                    {
                      "id": 14307913,
                      "name": "Marketing Website",
                      "code": "MW",
                      "is_active": true,
                      "bill_by": "Project",
                      "budget": 50,
                      "budget_by": "project",
                      "budget_is_monthly": false,
                      "notify_when_over_budget": true,
                      "over_budget_notification_percentage": 80,
                      "over_budget_notification_date": null,
                      "show_budget_to_all": false,
                      "created_at": "2017-06-26T21:36:23Z",
                      "updated_at": "2017-06-26T21:54:46Z",
                      "starts_on": "2017-01-01",
                      "ends_on": "2017-03-31",
                      "is_billable": true,
                      "is_fixed_fee": false,
                      "notes": "",
                      "client": {
                        "id": 5735774,
                        "name": "ABC Corp",
                        "currency": "USD"
                      },
                      "cost_budget": null,
                      "cost_budget_include_expenses": false,
                      "hourly_rate": 100,
                      "fee": null
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/projects?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/projects?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "is_active",
            "description": "Pass true to only return active projects and false to return inactive projects.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "client_id",
            "description": "Only return projects belonging to the client with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return projects that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a project",
        "operationId": "createProject",
        "description": "Creates a new project object. Returns a project object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a project",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/projects/#create-a-project"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a project",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Project"
                },
                "example": {
                  "id": 14308112,
                  "name": "Your New Project",
                  "code": null,
                  "is_active": true,
                  "bill_by": "Project",
                  "budget": 10000,
                  "budget_by": "project",
                  "budget_is_monthly": false,
                  "notify_when_over_budget": false,
                  "over_budget_notification_percentage": 80,
                  "over_budget_notification_date": null,
                  "show_budget_to_all": false,
                  "created_at": "2017-06-26T21:56:52Z",
                  "updated_at": "2017-06-26T21:56:52Z",
                  "starts_on": null,
                  "ends_on": null,
                  "is_billable": true,
                  "is_fixed_fee": false,
                  "notes": "",
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries",
                    "currency": "EUR"
                  },
                  "cost_budget": null,
                  "cost_budget_include_expenses": false,
                  "hourly_rate": 100,
                  "fee": null
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client to associate this project with.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "name": {
                    "type": "string",
                    "description": "The name of the project.",
                    "nullable": true
                  },
                  "code": {
                    "type": "string",
                    "description": "The code associated with the project.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the project is active or archived. Defaults to true.",
                    "nullable": true
                  },
                  "is_billable": {
                    "type": "boolean",
                    "description": "Whether the project is billable or not.",
                    "nullable": true
                  },
                  "is_fixed_fee": {
                    "type": "boolean",
                    "description": "Whether the project is a fixed-fee project or not.",
                    "nullable": true
                  },
                  "bill_by": {
                    "type": "string",
                    "description": "The method by which the project is invoiced. Options: Project, Tasks, People, or none.",
                    "nullable": true
                  },
                  "hourly_rate": {
                    "type": "number",
                    "description": "Rate for projects billed by Project Hourly Rate.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget": {
                    "type": "number",
                    "description": "The budget in hours for the project when budgeting by time.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget_by": {
                    "type": "string",
                    "description": "The method by which the project is budgeted. Options: project (Hours Per Project), project_cost (Total Project Fees), task (Hours Per Task), task_fees (Fees Per Task), person (Hours Per Person), none (No Budget).",
                    "nullable": true
                  },
                  "budget_is_monthly": {
                    "type": "boolean",
                    "description": "Option to have the budget reset every month. Defaults to false.",
                    "nullable": true
                  },
                  "notify_when_over_budget": {
                    "type": "boolean",
                    "description": "Whether Project Managers should be notified when the project goes over budget. Defaults to false.",
                    "nullable": true
                  },
                  "over_budget_notification_percentage": {
                    "type": "number",
                    "description": "Percentage value used to trigger over budget email alerts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "show_budget_to_all": {
                    "type": "boolean",
                    "description": "Option to show project budget to all employees. Does not apply to Total Project Fee projects. Defaults to false.",
                    "nullable": true
                  },
                  "cost_budget": {
                    "type": "number",
                    "description": "The monetary budget for the project when budgeting by money.",
                    "nullable": true,
                    "format": "float"
                  },
                  "cost_budget_include_expenses": {
                    "type": "boolean",
                    "description": "Option for budget of Total Project Fees projects to include tracked expenses. Defaults to false.",
                    "nullable": true
                  },
                  "fee": {
                    "type": "number",
                    "description": "The amount you plan to invoice for the project. Only used by fixed-fee projects.",
                    "nullable": true,
                    "format": "float"
                  },
                  "notes": {
                    "type": "string",
                    "description": "Project notes.",
                    "nullable": true
                  },
                  "starts_on": {
                    "type": "string",
                    "description": "Date the project was started.",
                    "nullable": true,
                    "format": "date"
                  },
                  "ends_on": {
                    "type": "string",
                    "description": "Date the project will end.",
                    "nullable": true,
                    "format": "date"
                  }
                },
                "required": [
                  "client_id",
                  "name",
                  "is_billable",
                  "bill_by",
                  "budget_by"
                ]
              }
            }
          }
        }
      }
    },
    "/projects/{projectId}": {
      "delete": {
        "summary": "Delete a project",
        "operationId": "deleteProject",
        "description": "Deletes a project and any time entries or expenses tracked to it.\nHowever, invoices associated with the project will not be deleted.\nIf you don’t want the project’s time entries and expenses to be deleted, you should archive the project instead.",
        "externalDocs": {
          "description": "Delete a project",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/projects/#delete-a-project"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a project"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a project",
        "operationId": "retrieveProject",
        "description": "Retrieves the project with the given ID. Returns a project object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a project",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/projects/#retrieve-a-project"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a project",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Project"
                },
                "example": {
                  "id": 14308069,
                  "name": "Online Store - Phase 1",
                  "code": "OS1",
                  "is_active": true,
                  "bill_by": "Project",
                  "budget": 200,
                  "budget_by": "project",
                  "budget_is_monthly": false,
                  "notify_when_over_budget": true,
                  "over_budget_notification_percentage": 80,
                  "over_budget_notification_date": null,
                  "show_budget_to_all": false,
                  "created_at": "2017-06-26T21:52:18Z",
                  "updated_at": "2017-06-26T21:54:06Z",
                  "starts_on": "2017-06-01",
                  "ends_on": null,
                  "is_billable": true,
                  "is_fixed_fee": false,
                  "notes": "",
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries",
                    "currency": "EUR"
                  },
                  "cost_budget": null,
                  "cost_budget_include_expenses": false,
                  "hourly_rate": 100,
                  "fee": null
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a project",
        "operationId": "updateProject",
        "description": "Updates the specific project by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a project object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a project",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/projects/#update-a-project"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a project",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Project"
                },
                "example": {
                  "id": 14308112,
                  "name": "New project name",
                  "code": null,
                  "is_active": true,
                  "bill_by": "Project",
                  "budget": 10000,
                  "budget_by": "project",
                  "budget_is_monthly": false,
                  "notify_when_over_budget": false,
                  "over_budget_notification_percentage": 80,
                  "over_budget_notification_date": null,
                  "show_budget_to_all": false,
                  "created_at": "2017-06-26T21:56:52Z",
                  "updated_at": "2017-06-26T21:57:20Z",
                  "starts_on": null,
                  "ends_on": null,
                  "is_billable": true,
                  "is_fixed_fee": false,
                  "notes": "",
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries",
                    "currency": "EUR"
                  },
                  "cost_budget": null,
                  "cost_budget_include_expenses": false,
                  "hourly_rate": 100,
                  "fee": null
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "client_id": {
                    "type": "integer",
                    "description": "The ID of the client to associate this project with.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "name": {
                    "type": "string",
                    "description": "The name of the project.",
                    "nullable": true
                  },
                  "code": {
                    "type": "string",
                    "description": "The code associated with the project.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the project is active or archived. Defaults to true.",
                    "nullable": true
                  },
                  "is_billable": {
                    "type": "boolean",
                    "description": "Whether the project is billable or not.",
                    "nullable": true
                  },
                  "is_fixed_fee": {
                    "type": "boolean",
                    "description": "Whether the project is a fixed-fee project or not.",
                    "nullable": true
                  },
                  "bill_by": {
                    "type": "string",
                    "description": "The method by which the project is invoiced. Options: Project, Tasks, People, or none.",
                    "nullable": true
                  },
                  "hourly_rate": {
                    "type": "number",
                    "description": "Rate for projects billed by Project Hourly Rate.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget": {
                    "type": "number",
                    "description": "The budget in hours for the project when budgeting by time.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget_by": {
                    "type": "string",
                    "description": "The method by which the project is budgeted. Options: project (Hours Per Project), project_cost (Total Project Fees), task (Hours Per Task), task_fees (Fees Per Task), person (Hours Per Person), none (No Budget).",
                    "nullable": true
                  },
                  "budget_is_monthly": {
                    "type": "boolean",
                    "description": "Option to have the budget reset every month. Defaults to false.",
                    "nullable": true
                  },
                  "notify_when_over_budget": {
                    "type": "boolean",
                    "description": "Whether Project Managers should be notified when the project goes over budget. Defaults to false.",
                    "nullable": true
                  },
                  "over_budget_notification_percentage": {
                    "type": "number",
                    "description": "Percentage value used to trigger over budget email alerts. Example: use 10.0 for 10.0%.",
                    "nullable": true,
                    "format": "float"
                  },
                  "show_budget_to_all": {
                    "type": "boolean",
                    "description": "Option to show project budget to all employees. Does not apply to Total Project Fee projects. Defaults to false.",
                    "nullable": true
                  },
                  "cost_budget": {
                    "type": "number",
                    "description": "The monetary budget for the project when budgeting by money.",
                    "nullable": true,
                    "format": "float"
                  },
                  "cost_budget_include_expenses": {
                    "type": "boolean",
                    "description": "Option for budget of Total Project Fees projects to include tracked expenses. Defaults to false.",
                    "nullable": true
                  },
                  "fee": {
                    "type": "number",
                    "description": "The amount you plan to invoice for the project. Only used by fixed-fee projects.",
                    "nullable": true,
                    "format": "float"
                  },
                  "notes": {
                    "type": "string",
                    "description": "Project notes.",
                    "nullable": true
                  },
                  "starts_on": {
                    "type": "string",
                    "description": "Date the project was started.",
                    "nullable": true,
                    "format": "date"
                  },
                  "ends_on": {
                    "type": "string",
                    "description": "Date the project will end.",
                    "nullable": true,
                    "format": "date"
                  }
                }
              }
            }
          }
        }
      }
    },
    "/projects/{projectId}/task_assignments": {
      "get": {
        "summary": "List all task assignments for a specific project",
        "operationId": "listTaskAssignmentsForSpecificProject",
        "description": "Returns a list of your task assignments for the project identified by PROJECT_ID. The task assignments are returned sorted by creation date, with the most recently created task assignments appearing first.\n\nThe response contains an object with a task_assignments property that contains an array of up to per_page task assignments. Each entry in the array is a separate task assignment object. If no more task assignments are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your task assignments.",
        "externalDocs": {
          "description": "List all task assignments for a specific project",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/task-assignments/#list-all-task-assignments-for-a-specific-project"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all task assignments for a specific project",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TaskAssignments"
                },
                "example": {
                  "task_assignments": [
                    {
                      "id": 155505016,
                      "billable": false,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:54:06Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083369,
                        "name": "Research"
                      }
                    },
                    {
                      "id": 155505015,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083368,
                        "name": "Project Management"
                      }
                    },
                    {
                      "id": 155505014,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083366,
                        "name": "Programming"
                      }
                    },
                    {
                      "id": 155505013,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083365,
                        "name": "Graphic Design"
                      }
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 4,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/projects/14308069/task_assignments?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/projects/14308069/task_assignments?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "is_active",
            "description": "Pass true to only return active task assignments and false to return inactive task assignments.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return task assignments that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a task assignment",
        "operationId": "createTaskAssignment",
        "description": "Creates a new task assignment object. Returns a task assignment object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a task assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/task-assignments/#create-a-task-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a task assignment",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TaskAssignment"
                },
                "example": {
                  "id": 155506339,
                  "billable": true,
                  "is_active": true,
                  "created_at": "2017-06-26T22:10:43Z",
                  "updated_at": "2017-06-26T22:10:43Z",
                  "hourly_rate": 75.5,
                  "budget": null,
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "task": {
                    "id": 8083800,
                    "name": "Business Development"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "task_id": {
                    "type": "integer",
                    "description": "The ID of the task to associate with the project.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the task assignment is active or archived. Defaults to true.",
                    "nullable": true
                  },
                  "billable": {
                    "type": "boolean",
                    "description": "Whether the task assignment is billable or not. Defaults to false.",
                    "nullable": true
                  },
                  "hourly_rate": {
                    "type": "number",
                    "description": "Rate used when the project’s bill_by is Tasks. Defaults to null when billing by task hourly rate, otherwise 0.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget": {
                    "type": "number",
                    "description": "Budget used when the project’s budget_by is task or task_fees.",
                    "nullable": true,
                    "format": "float"
                  }
                },
                "required": [
                  "task_id"
                ]
              }
            }
          }
        }
      }
    },
    "/projects/{projectId}/task_assignments/{taskAssignmentId}": {
      "delete": {
        "summary": "Delete a task assignment",
        "operationId": "deleteTaskAssignment",
        "description": "Delete a task assignment. Deleting a task assignment is only possible if it has no time entries associated with it. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a task assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/task-assignments/#delete-a-task-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a task assignment"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "taskAssignmentId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a task assignment",
        "operationId": "retrieveTaskAssignment",
        "description": "Retrieves the task assignment with the given ID. Returns a task assignment object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a task assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/task-assignments/#retrieve-a-task-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a task assignment",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TaskAssignment"
                },
                "example": {
                  "id": 155505016,
                  "billable": false,
                  "is_active": true,
                  "created_at": "2017-06-26T21:52:18Z",
                  "updated_at": "2017-06-26T21:54:06Z",
                  "hourly_rate": 100,
                  "budget": null,
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "task": {
                    "id": 8083369,
                    "name": "Research"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "taskAssignmentId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a task assignment",
        "operationId": "updateTaskAssignment",
        "description": "Updates the specific task assignment by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a task assignment object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a task assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/task-assignments/#update-a-task-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a task assignment",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TaskAssignment"
                },
                "example": {
                  "id": 155506339,
                  "billable": true,
                  "is_active": true,
                  "created_at": "2017-06-26T22:10:43Z",
                  "updated_at": "2017-06-26T22:11:27Z",
                  "hourly_rate": 75.5,
                  "budget": 120,
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "task": {
                    "id": 8083800,
                    "name": "Business Development"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "taskAssignmentId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the task assignment is active or archived.",
                    "nullable": true
                  },
                  "billable": {
                    "type": "boolean",
                    "description": "Whether the task assignment is billable or not.",
                    "nullable": true
                  },
                  "hourly_rate": {
                    "type": "number",
                    "description": "Rate used when the project’s bill_by is Tasks.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget": {
                    "type": "number",
                    "description": "Budget used when the project’s budget_by is task or task_fees.",
                    "nullable": true,
                    "format": "float"
                  }
                }
              }
            }
          }
        }
      }
    },
    "/projects/{projectId}/user_assignments": {
      "get": {
        "summary": "List all user assignments for a specific project",
        "operationId": "listUserAssignmentsForSpecificProject",
        "description": "Returns a list of user assignments for the project identified by PROJECT_ID. The user assignments are returned sorted by creation date, with the most recently created user assignments appearing first.\n\nThe response contains an object with a user_assignments property that contains an array of up to per_page user assignments. Each entry in the array is a separate user assignment object. If no more user assignments are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your user assignments.",
        "externalDocs": {
          "description": "List all user assignments for a specific project",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/user-assignments/#list-all-user-assignments-for-a-specific-project"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all user assignments for a specific project",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UserAssignments"
                },
                "example": {
                  "user_assignments": [
                    {
                      "id": 125068554,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": true,
                      "budget": null,
                      "created_at": "2017-06-26T22:32:52Z",
                      "updated_at": "2017-06-26T22:32:52Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      }
                    },
                    {
                      "id": 125066109,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": false,
                      "budget": null,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "user": {
                        "id": 1782884,
                        "name": "Jeremy Israelsen"
                      }
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/projects/14308069/user_assignments?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/projects/14308069/user_assignments?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "user_id",
            "description": "Only return user assignments belonging to the user with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "is_active",
            "description": "Pass true to only return active user assignments and false to return inactive user assignments.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return user assignments that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a user assignment",
        "operationId": "createUserAssignment",
        "description": "Creates a new user assignment object. Returns a user assignment object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a user assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/user-assignments/#create-a-user-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a user assignment",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UserAssignment"
                },
                "example": {
                  "id": 125068758,
                  "is_project_manager": false,
                  "is_active": true,
                  "use_default_rates": false,
                  "budget": null,
                  "created_at": "2017-06-26T22:36:01Z",
                  "updated_at": "2017-06-26T22:36:01Z",
                  "hourly_rate": 75.5,
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "user": {
                    "id": 1782974,
                    "name": "Jim Allen"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "user_id": {
                    "type": "integer",
                    "description": "The ID of the user to associate with the project.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the user assignment is active or archived. Defaults to true.",
                    "nullable": true
                  },
                  "is_project_manager": {
                    "type": "boolean",
                    "description": "Determines if the user has Project Manager permissions for the project. Defaults to false for users with Regular User permissions and true for those with Project Managers or Administrator permissions.",
                    "nullable": true
                  },
                  "use_default_rates": {
                    "type": "boolean",
                    "description": "Determines which billable rate(s) will be used on the project for this user when bill_by is People. When true, the project will use the user’s default billable rates. When false, the project will use the custom rate defined on this user assignment. Defaults to true.",
                    "nullable": true
                  },
                  "hourly_rate": {
                    "type": "number",
                    "description": "Custom rate used when the project’s bill_by is People and use_default_rates is false. Defaults to 0.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget": {
                    "type": "number",
                    "description": "Budget used when the project’s budget_by is person.",
                    "nullable": true,
                    "format": "float"
                  }
                },
                "required": [
                  "user_id"
                ]
              }
            }
          }
        }
      }
    },
    "/projects/{projectId}/user_assignments/{userAssignmentId}": {
      "delete": {
        "summary": "Delete a user assignment",
        "operationId": "deleteUserAssignment",
        "description": "Delete a user assignment. Deleting a user assignment is only possible if it has no time entries or expenses associated with it. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a user assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/user-assignments/#delete-a-user-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a user assignment"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "userAssignmentId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a user assignment",
        "operationId": "retrieveUserAssignment",
        "description": "Retrieves the user assignment with the given ID. Returns a user assignment object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a user assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/user-assignments/#retrieve-a-user-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a user assignment",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UserAssignment"
                },
                "example": {
                  "id": 125068554,
                  "is_project_manager": true,
                  "is_active": true,
                  "use_default_rates": true,
                  "budget": null,
                  "created_at": "2017-06-26T22:32:52Z",
                  "updated_at": "2017-06-26T22:32:52Z",
                  "hourly_rate": 100,
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "user": {
                    "id": 1782959,
                    "name": "Kim Allen"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "userAssignmentId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a user assignment",
        "operationId": "updateUserAssignment",
        "description": "Updates the specific user assignment by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a user assignment object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a user assignment",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/user-assignments/#update-a-user-assignment"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a user assignment",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UserAssignment"
                },
                "example": {
                  "id": 125068758,
                  "is_project_manager": false,
                  "is_active": true,
                  "use_default_rates": false,
                  "budget": 120,
                  "created_at": "2017-06-26T22:36:01Z",
                  "updated_at": "2017-06-26T22:36:35Z",
                  "hourly_rate": 75.5,
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1",
                    "code": "OS1"
                  },
                  "user": {
                    "id": 1782974,
                    "name": "Jim Allen"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "projectId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "userAssignmentId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the user assignment is active or archived.",
                    "nullable": true
                  },
                  "is_project_manager": {
                    "type": "boolean",
                    "description": "Determines if the user has Project Manager permissions for the project.",
                    "nullable": true
                  },
                  "use_default_rates": {
                    "type": "boolean",
                    "description": "Determines which billable rate(s) will be used on the project for this user when bill_by is People. When true, the project will use the user’s default billable rates. When false, the project will use the custom rate defined on this user assignment.",
                    "nullable": true
                  },
                  "hourly_rate": {
                    "type": "number",
                    "description": "Custom rate used when the project’s bill_by is People and use_default_rates is false.",
                    "nullable": true,
                    "format": "float"
                  },
                  "budget": {
                    "type": "number",
                    "description": "Budget used when the project’s budget_by is person.",
                    "nullable": true,
                    "format": "float"
                  }
                }
              }
            }
          }
        }
      }
    },
    "/reports/expenses/categories": {
      "get": {
        "summary": "Expense Categories Report",
        "operationId": "expenseCategoriesReport",
        "description": "",
        "externalDocs": {
          "description": "Expense Categories Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/expense-reports/#expense-categories-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Expense Categories Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "expense_category_id": 4197501,
                      "expense_category_name": "Lodging",
                      "total_amount": 100,
                      "billable_amount": 100,
                      "currency": "EUR"
                    },
                    {
                      "expense_category_id": 4195926,
                      "expense_category_name": "Meals",
                      "total_amount": 100,
                      "billable_amount": 100,
                      "currency": "EUR"
                    },
                    {
                      "expense_category_id": 4195926,
                      "expense_category_name": "Meals",
                      "total_amount": 33.35,
                      "billable_amount": 33.35,
                      "currency": "USD"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 3,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/expenses/categories?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/expenses/categories?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on expenses with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on expenses with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/expenses/clients": {
      "get": {
        "summary": "Clients Report",
        "operationId": "clientsExpensesReport",
        "description": "",
        "externalDocs": {
          "description": "Clients Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/expense-reports/#clients-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Clients Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "total_amount": 100,
                      "billable_amount": 100,
                      "currency": "EUR"
                    },
                    {
                      "client_id": 5735774,
                      "client_name": "ABC Corp",
                      "total_amount": 133.35,
                      "billable_amount": 133.35,
                      "currency": "USD"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/expenses/clients?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/expenses/clients?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on expenses with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on expenses with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/expenses/projects": {
      "get": {
        "summary": "Projects Report",
        "operationId": "projectsExpensesReport",
        "description": "",
        "externalDocs": {
          "description": "Projects Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/expense-reports/#projects-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Projects Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "client_id": 5735774,
                      "client_name": "ABC Corp",
                      "project_id": 14307913,
                      "project_name": "[MW] Marketing Website",
                      "total_amount": 133.35,
                      "billable_amount": 133.35,
                      "currency": "USD"
                    },
                    {
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "project_id": 14308069,
                      "project_name": "[OS1] Online Store - Phase 1",
                      "total_amount": 100,
                      "billable_amount": 100,
                      "currency": "EUR"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/expenses/projects?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/expenses/projects?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on expenses with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on expenses with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/expenses/team": {
      "get": {
        "summary": "Team Report",
        "operationId": "teamExpensesReport",
        "description": "",
        "externalDocs": {
          "description": "Team Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/expense-reports/#team-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Team Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ExpenseReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "user_id": 1782884,
                      "user_name": "Bob Powell",
                      "is_contractor": false,
                      "total_amount": 100,
                      "billable_amount": 100,
                      "currency": "USD"
                    },
                    {
                      "user_id": 1782959,
                      "user_name": "Kim Allen",
                      "is_contractor": false,
                      "total_amount": 100,
                      "billable_amount": 100,
                      "currency": "EUR"
                    },
                    {
                      "user_id": 1782959,
                      "user_name": "Kim Allen",
                      "is_contractor": false,
                      "total_amount": 33.35,
                      "billable_amount": 33.35,
                      "currency": "USD"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 3,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/expenses/team?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/expenses/team?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on expenses with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on expenses with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/project_budget": {
      "get": {
        "summary": "Project Budget Report",
        "operationId": "projectBudgetReport",
        "description": "The response contains an object with a results property that contains an array of up to per_page results. Each entry in the array is a separate result object. If no more results are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your results.",
        "externalDocs": {
          "description": "Project Budget Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/project-budget-report/#project-budget-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Project Budget Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ProjectBudgetReportResults"
                },
                "example": {
                  "results": [
                    {
                      "project_id": 14308069,
                      "project_name": "Online Store - Phase 1",
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "budget_is_monthly": false,
                      "budget_by": "project",
                      "is_active": true,
                      "budget": 200,
                      "budget_spent": 4,
                      "budget_remaining": 196
                    },
                    {
                      "project_id": 14307913,
                      "project_name": "Marketing Website",
                      "client_id": 5735774,
                      "client_name": "ABC Corp",
                      "budget_is_monthly": false,
                      "budget_by": "project",
                      "is_active": true,
                      "budget": 50,
                      "budget_spent": 2,
                      "budget_remaining": 48
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/project_budget?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/project_budget?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "is_active",
            "description": "Pass true to only return active projects and false to return inactive projects.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          }
        ]
      }
    },
    "/reports/time/clients": {
      "get": {
        "summary": "Clients Report",
        "operationId": "clientsTimeReport",
        "description": "",
        "externalDocs": {
          "description": "Clients Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/time-reports/#clients-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Clients Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "total_hours": 4.5,
                      "billable_hours": 3.5,
                      "currency": "EUR",
                      "billable_amount": 350
                    },
                    {
                      "client_id": 5735774,
                      "client_name": "ABC Corp",
                      "total_hours": 2,
                      "billable_hours": 2,
                      "currency": "USD",
                      "billable_amount": 200
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/time/clients?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/time/clients?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on time entries with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on time entries with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "include_fixed_fee",
            "description": "When true, billable amounts will be calculated and included for fixed fee projects.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/time/projects": {
      "get": {
        "summary": "Projects Report",
        "operationId": "projectsTimeReport",
        "description": "",
        "externalDocs": {
          "description": "Projects Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/time-reports/#projects-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Projects Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "project_id": 14307913,
                      "project_name": "[MW] Marketing Website",
                      "client_id": 5735774,
                      "client_name": "ABC Corp",
                      "total_hours": 2,
                      "billable_hours": 2,
                      "currency": "USD",
                      "billable_amount": 200
                    },
                    {
                      "project_id": 14308069,
                      "project_name": "[OS1] Online Store - Phase 1",
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "total_hours": 4,
                      "billable_hours": 3,
                      "currency": "EUR",
                      "billable_amount": 300
                    },
                    {
                      "project_id": 14808188,
                      "project_name": "[TF] Task Force",
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "total_hours": 0.5,
                      "billable_hours": 0.5,
                      "currency": "EUR",
                      "billable_amount": 50
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 3,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/time/projects?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/time/projects?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on time entries with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on time entries with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "include_fixed_fee",
            "description": "When true, billable amounts will be calculated and included for fixed fee projects.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/time/tasks": {
      "get": {
        "summary": "Tasks Report",
        "operationId": "tasksReport",
        "description": "",
        "externalDocs": {
          "description": "Tasks Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/time-reports/#tasks-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Tasks Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "task_id": 8083365,
                      "task_name": "Graphic Design",
                      "total_hours": 2,
                      "billable_hours": 2,
                      "currency": "USD",
                      "billable_amount": 200
                    },
                    {
                      "task_id": 8083366,
                      "task_name": "Programming",
                      "total_hours": 1.5,
                      "billable_hours": 1.5,
                      "currency": "EUR",
                      "billable_amount": 150
                    },
                    {
                      "task_id": 8083368,
                      "task_name": "Project Management",
                      "total_hours": 1.5,
                      "billable_hours": 1.5,
                      "currency": "EUR",
                      "billable_amount": 150
                    },
                    {
                      "task_id": 8083368,
                      "task_name": "Project Management",
                      "total_hours": 0.5,
                      "billable_hours": 0.5,
                      "currency": "USD",
                      "billable_amount": 50
                    },
                    {
                      "task_id": 8083369,
                      "task_name": "Research",
                      "total_hours": 1,
                      "billable_hours": 0,
                      "currency": "EUR",
                      "billable_amount": 0
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 5,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/time/tasks?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/time/tasks?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on time entries with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on time entries with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "include_fixed_fee",
            "description": "When true, billable amounts will be calculated and included for fixed fee projects.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/time/team": {
      "get": {
        "summary": "Team Report",
        "operationId": "teamTimeReport",
        "description": "",
        "externalDocs": {
          "description": "Team Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/time-reports/#team-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Team Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeReportsResults"
                },
                "example": {
                  "results": [
                    {
                      "user_id": 1795925,
                      "user_name": "Jane Smith",
                      "is_contractor": false,
                      "total_hours": 0.5,
                      "billable_hours": 0.5,
                      "currency": "EUR",
                      "billable_amount": 50,
                      "weekly_capacity": 126000,
                      "avatar_url": "https://cache.harvestapp.com/assets/profile_images/abraj_albait_towers.png?1498516481"
                    },
                    {
                      "user_id": 1782959,
                      "user_name": "Kim Allen",
                      "is_contractor": false,
                      "total_hours": 4,
                      "billable_hours": 3,
                      "currency": "EUR",
                      "billable_amount": 300,
                      "weekly_capacity": 126000,
                      "avatar_url": "https://cache.harvestapp.com/assets/profile_images/cornell_clock_tower.png?1498515345"
                    },
                    {
                      "user_id": 1782959,
                      "user_name": "Kim Allen",
                      "is_contractor": false,
                      "total_hours": 2,
                      "billable_hours": 2,
                      "currency": "USD",
                      "billable_amount": 200,
                      "weekly_capacity": 126000,
                      "avatar_url": "https://cache.harvestapp.com/assets/profile_images/allen_bradley_clock_tower.png?1498509661"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 3,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/time/team?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/time/team?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on time entries with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on time entries with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "include_fixed_fee",
            "description": "When true, billable amounts will be calculated and included for fixed fee projects.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/reports/uninvoiced": {
      "get": {
        "summary": "Uninvoiced Report",
        "operationId": "uninvoicedReport",
        "description": "The response contains an object with a results property that contains an array of up to per_page results. Each entry in the array is a separate result object. If no more results are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your results.\n\nNote: Each request requires both the from and to parameters to be supplied in the URL’s query string. The timeframe supplied cannot exceed 1 year (365 days).",
        "externalDocs": {
          "description": "Uninvoiced Report",
          "url": "https://help.getharvest.com/api-v2/reports-api/reports/uninvoiced-report/#uninvoiced-report"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Uninvoiced Report",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UninvoicedReportResults"
                },
                "example": {
                  "results": [
                    {
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "project_id": 14308069,
                      "project_name": "Online Store - Phase 1",
                      "currency": "EUR",
                      "total_hours": 4,
                      "uninvoiced_hours": 0,
                      "uninvoiced_expenses": 100,
                      "uninvoiced_amount": 100
                    },
                    {
                      "client_id": 5735776,
                      "client_name": "123 Industries",
                      "project_id": 14808188,
                      "project_name": "Task Force",
                      "currency": "EUR",
                      "total_hours": 0.5,
                      "uninvoiced_hours": 0.5,
                      "uninvoiced_expenses": 0,
                      "uninvoiced_amount": 50
                    },
                    {
                      "client_id": 5735774,
                      "client_name": "ABC Corp",
                      "project_id": 14307913,
                      "project_name": "Marketing Website",
                      "currency": "USD",
                      "total_hours": 2,
                      "uninvoiced_hours": 0,
                      "uninvoiced_expenses": 0,
                      "uninvoiced_amount": 0
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 3,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/reports/uninvoiced?from=20170101&page=1&per_page=2000&to=20171231",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/reports/uninvoiced?from=20170101&page=1&per_page=2000&to=20171231"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "from",
            "description": "Only report on time entries and expenses with a spent_date on or after the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only report on time entries and expenses with a spent_date on or before the given date.",
            "required": true,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/roles": {
      "get": {
        "summary": "List all roles",
        "operationId": "listRoles",
        "description": "Returns a list of roles in the account. The roles are returned sorted by creation date, with the most recently created roles appearing first.\n\nThe response contains an object with a roles property that contains an array of up to per_page roles. Each entry in the array is a separate role object. If no more roles are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your roles.",
        "externalDocs": {
          "description": "List all roles",
          "url": "https://help.getharvest.com/api-v2/roles-api/roles/roles/#list-all-roles"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all roles",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Roles"
                },
                "example": {
                  "roles": [
                    {
                      "id": 618100,
                      "name": "Designer",
                      "created_at": "2020-04-17T10:09:41Z",
                      "updated_at": "2020-04-17T10:09:41Z",
                      "user_ids": []
                    },
                    {
                      "id": 618099,
                      "name": "Developer",
                      "created_at": "2020-04-17T10:08:43Z",
                      "updated_at": "2020-04-17T10:08:43Z",
                      "user_ids": []
                    },
                    {
                      "id": 617630,
                      "name": "Sales",
                      "created_at": "2020-04-16T16:59:59Z",
                      "updated_at": "2020-04-16T16:59:59Z",
                      "user_ids": [
                        2084359,
                        3122373,
                        3122374
                      ]
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/roles?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/roles?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a role",
        "operationId": "createRole",
        "description": "Creates a new role object. Returns a role object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a role",
          "url": "https://help.getharvest.com/api-v2/roles-api/roles/roles/#create-a-role"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a role",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Role"
                },
                "example": {
                  "id": 617670,
                  "name": "Marketing",
                  "created_at": "2020-04-16T18:18:30Z",
                  "updated_at": "2020-04-16T18:18:30Z",
                  "user_ids": [
                    3122374,
                    3122373,
                    2084359
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the role.",
                    "nullable": true
                  },
                  "user_ids": {
                    "type": "array",
                    "description": "The IDs of the users assigned to this role.",
                    "nullable": true,
                    "items": {
                      "type": "integer"
                    }
                  }
                },
                "required": [
                  "name"
                ]
              }
            }
          }
        }
      }
    },
    "/roles/{roleId}": {
      "delete": {
        "summary": "Delete a role",
        "operationId": "deleteRole",
        "description": "Delete a role. Deleting a role will unlink it from any users it was assigned to. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a role",
          "url": "https://help.getharvest.com/api-v2/roles-api/roles/roles/#delete-a-role"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a role"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "roleId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a role",
        "operationId": "retrieveRole",
        "description": "Retrieves the role with the given ID. Returns a role object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a role",
          "url": "https://help.getharvest.com/api-v2/roles-api/roles/roles/#retrieve-a-role"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a role",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Role"
                },
                "example": {
                  "id": 617630,
                  "name": "Sales",
                  "created_at": "2020-04-16T16:59:59Z",
                  "updated_at": "2020-04-16T16:59:59Z",
                  "user_ids": [
                    2084359,
                    3122373,
                    3122374
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "roleId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a role",
        "operationId": "updateRole",
        "description": "Updates the specific role by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a role object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a role",
          "url": "https://help.getharvest.com/api-v2/roles-api/roles/roles/#update-a-role"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a role",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Role"
                },
                "example": {
                  "id": 618099,
                  "name": "HR",
                  "created_at": "2020-04-16T17:00:38Z",
                  "updated_at": "2020-04-16T17:00:57Z",
                  "user_ids": [
                    2084359
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "roleId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the role.",
                    "nullable": true
                  },
                  "user_ids": {
                    "type": "array",
                    "description": "The IDs of the users assigned to this role.",
                    "nullable": true,
                    "items": {
                      "type": "integer"
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/task_assignments": {
      "get": {
        "summary": "List all task assignments",
        "operationId": "listTaskAssignments",
        "description": "Returns a list of your task assignments. The task assignments are returned sorted by creation date, with the most recently created task assignments appearing first.\n\nThe response contains an object with a task_assignments property that contains an array of up to per_page task assignments. Each entry in the array is a separate task assignment object. If no more task assignments are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your task assignments.",
        "externalDocs": {
          "description": "List all task assignments",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/task-assignments/#list-all-task-assignments"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all task assignments",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TaskAssignments"
                },
                "example": {
                  "task_assignments": [
                    {
                      "id": 160726647,
                      "billable": false,
                      "is_active": true,
                      "created_at": "2017-08-22T17:36:54Z",
                      "updated_at": "2017-08-22T17:36:54Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14808188,
                        "name": "Task Force",
                        "code": "TF"
                      },
                      "task": {
                        "id": 8083369,
                        "name": "Research"
                      }
                    },
                    {
                      "id": 160726646,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-08-22T17:36:54Z",
                      "updated_at": "2017-08-22T17:36:54Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14808188,
                        "name": "Task Force",
                        "code": "TF"
                      },
                      "task": {
                        "id": 8083368,
                        "name": "Project Management"
                      }
                    },
                    {
                      "id": 160726645,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-08-22T17:36:54Z",
                      "updated_at": "2017-08-22T17:36:54Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14808188,
                        "name": "Task Force",
                        "code": "TF"
                      },
                      "task": {
                        "id": 8083366,
                        "name": "Programming"
                      }
                    },
                    {
                      "id": 160726644,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-08-22T17:36:54Z",
                      "updated_at": "2017-08-22T17:36:54Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14808188,
                        "name": "Task Force",
                        "code": "TF"
                      },
                      "task": {
                        "id": 8083365,
                        "name": "Graphic Design"
                      }
                    },
                    {
                      "id": 155505153,
                      "billable": false,
                      "is_active": true,
                      "created_at": "2017-06-26T21:53:20Z",
                      "updated_at": "2017-06-26T21:54:31Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "task": {
                        "id": 8083369,
                        "name": "Research"
                      }
                    },
                    {
                      "id": 155505016,
                      "billable": false,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:54:06Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083369,
                        "name": "Research"
                      }
                    },
                    {
                      "id": 155505015,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083368,
                        "name": "Project Management"
                      }
                    },
                    {
                      "id": 155505014,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083366,
                        "name": "Programming"
                      }
                    },
                    {
                      "id": 155505013,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "task": {
                        "id": 8083365,
                        "name": "Graphic Design"
                      }
                    },
                    {
                      "id": 155502711,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:36:23Z",
                      "updated_at": "2017-06-26T21:36:23Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "task": {
                        "id": 8083368,
                        "name": "Project Management"
                      }
                    },
                    {
                      "id": 155502710,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:36:23Z",
                      "updated_at": "2017-06-26T21:36:23Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "task": {
                        "id": 8083366,
                        "name": "Programming"
                      }
                    },
                    {
                      "id": 155502709,
                      "billable": true,
                      "is_active": true,
                      "created_at": "2017-06-26T21:36:23Z",
                      "updated_at": "2017-06-26T21:36:23Z",
                      "hourly_rate": 100,
                      "budget": null,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "task": {
                        "id": 8083365,
                        "name": "Graphic Design"
                      }
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 12,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/task_assignments?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/task_assignments?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "is_active",
            "description": "Pass true to only return active task assignments and false to return inactive task assignments.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return task assignments that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/tasks": {
      "get": {
        "summary": "List all tasks",
        "operationId": "listTasks",
        "description": "Returns a list of your tasks. The tasks are returned sorted by creation date, with the most recently created tasks appearing first.\n\nThe response contains an object with a tasks property that contains an array of up to per_page tasks. Each entry in the array is a separate task object. If no more tasks are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your tasks.",
        "externalDocs": {
          "description": "List all tasks",
          "url": "https://help.getharvest.com/api-v2/tasks-api/tasks/tasks/#list-all-tasks"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all tasks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Tasks"
                },
                "example": {
                  "tasks": [
                    {
                      "id": 8083800,
                      "name": "Business Development",
                      "billable_by_default": false,
                      "default_hourly_rate": 0,
                      "is_default": false,
                      "is_active": true,
                      "created_at": "2017-06-26T22:08:25Z",
                      "updated_at": "2017-06-26T22:08:25Z"
                    },
                    {
                      "id": 8083369,
                      "name": "Research",
                      "billable_by_default": false,
                      "default_hourly_rate": 0,
                      "is_default": true,
                      "is_active": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T21:53:34Z"
                    },
                    {
                      "id": 8083368,
                      "name": "Project Management",
                      "billable_by_default": true,
                      "default_hourly_rate": 100,
                      "is_default": true,
                      "is_active": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T21:14:10Z"
                    },
                    {
                      "id": 8083366,
                      "name": "Programming",
                      "billable_by_default": true,
                      "default_hourly_rate": 100,
                      "is_default": true,
                      "is_active": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T21:14:07Z"
                    },
                    {
                      "id": 8083365,
                      "name": "Graphic Design",
                      "billable_by_default": true,
                      "default_hourly_rate": 100,
                      "is_default": true,
                      "is_active": true,
                      "created_at": "2017-06-26T20:41:00Z",
                      "updated_at": "2017-06-26T21:14:02Z"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 5,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/tasks?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/tasks?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "is_active",
            "description": "Pass true to only return active tasks and false to return inactive tasks.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return tasks that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a task",
        "operationId": "createTask",
        "description": "Creates a new task object. Returns a task object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a task",
          "url": "https://help.getharvest.com/api-v2/tasks-api/tasks/tasks/#create-a-task"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a task",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                },
                "example": {
                  "id": 8083782,
                  "name": "New Task Name",
                  "billable_by_default": true,
                  "default_hourly_rate": 0,
                  "is_default": false,
                  "is_active": true,
                  "created_at": "2017-06-26T22:04:31Z",
                  "updated_at": "2017-06-26T22:04:31Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the task.",
                    "nullable": true
                  },
                  "billable_by_default": {
                    "type": "boolean",
                    "description": "Used in determining whether default tasks should be marked billable when creating a new project. Defaults to true.",
                    "nullable": true
                  },
                  "default_hourly_rate": {
                    "type": "number",
                    "description": "The default hourly rate to use for this task when it is added to a project. Defaults to 0.",
                    "nullable": true,
                    "format": "float"
                  },
                  "is_default": {
                    "type": "boolean",
                    "description": "Whether this task should be automatically added to future projects. Defaults to false.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether this task is active or archived. Defaults to true.",
                    "nullable": true
                  }
                },
                "required": [
                  "name"
                ]
              }
            }
          }
        }
      }
    },
    "/tasks/{taskId}": {
      "delete": {
        "summary": "Delete a task",
        "operationId": "deleteTask",
        "description": "Delete a task. Deleting a task is only possible if it has no time entries associated with it. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a task",
          "url": "https://help.getharvest.com/api-v2/tasks-api/tasks/tasks/#delete-a-task"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a task"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "taskId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a task",
        "operationId": "retrieveTask",
        "description": "Retrieves the task with the given ID. Returns a task object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a task",
          "url": "https://help.getharvest.com/api-v2/tasks-api/tasks/tasks/#retrieve-a-task"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a task",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                },
                "example": {
                  "id": 8083800,
                  "name": "Business Development",
                  "billable_by_default": false,
                  "default_hourly_rate": 0,
                  "is_default": false,
                  "is_active": true,
                  "created_at": "2017-06-26T22:08:25Z",
                  "updated_at": "2017-06-26T22:08:25Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "taskId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a task",
        "operationId": "updateTask",
        "description": "Updates the specific task by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a task object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a task",
          "url": "https://help.getharvest.com/api-v2/tasks-api/tasks/tasks/#update-a-task"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a task",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                },
                "example": {
                  "id": 8083782,
                  "name": "New Task Name",
                  "billable_by_default": true,
                  "default_hourly_rate": 0,
                  "is_default": true,
                  "is_active": true,
                  "created_at": "2017-06-26T22:04:31Z",
                  "updated_at": "2017-06-26T22:04:54Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "taskId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string",
                    "description": "The name of the task.",
                    "nullable": true
                  },
                  "billable_by_default": {
                    "type": "boolean",
                    "description": "Used in determining whether default tasks should be marked billable when creating a new project.",
                    "nullable": true
                  },
                  "default_hourly_rate": {
                    "type": "number",
                    "description": "The default hourly rate to use for this task when it is added to a project.",
                    "nullable": true,
                    "format": "float"
                  },
                  "is_default": {
                    "type": "boolean",
                    "description": "Whether this task should be automatically added to future projects.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether this task is active or archived.",
                    "nullable": true
                  }
                }
              }
            }
          }
        }
      }
    },
    "/time_entries": {
      "get": {
        "summary": "List all time entries",
        "operationId": "listTimeEntries",
        "description": "Returns a list of time entries. The time entries are returned sorted by spent_date date. At this time, the sort option can’t be customized.\n\nThe response contains an object with a time_entries property that contains an array of up to per_page time entries. Each entry in the array is a separate time entry object. If no more time entries are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your time entries.",
        "externalDocs": {
          "description": "List all time entries",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#list-all-time-entries"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all time entries",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeEntries"
                },
                "example": {
                  "time_entries": [
                    {
                      "id": 636709355,
                      "spent_date": "2017-03-02",
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      },
                      "client": {
                        "id": 5735774,
                        "name": "ABC Corp"
                      },
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website"
                      },
                      "task": {
                        "id": 8083365,
                        "name": "Graphic Design"
                      },
                      "user_assignment": {
                        "id": 125068553,
                        "is_project_manager": true,
                        "is_active": true,
                        "budget": null,
                        "created_at": "2017-06-26T22:32:52Z",
                        "updated_at": "2017-06-26T22:32:52Z",
                        "hourly_rate": 100
                      },
                      "task_assignment": {
                        "id": 155502709,
                        "billable": true,
                        "is_active": true,
                        "created_at": "2017-06-26T21:36:23Z",
                        "updated_at": "2017-06-26T21:36:23Z",
                        "hourly_rate": 100,
                        "budget": null
                      },
                      "hours": 2.11,
                      "hours_without_timer": 2.11,
                      "rounded_hours": 2.25,
                      "notes": "Adding CSS styling",
                      "created_at": "2017-06-27T15:50:15Z",
                      "updated_at": "2017-06-27T16:47:14Z",
                      "is_locked": true,
                      "locked_reason": "Item Approved and Locked for this Time Period",
                      "is_closed": true,
                      "is_billed": false,
                      "timer_started_at": null,
                      "started_time": "3:00pm",
                      "ended_time": "5:00pm",
                      "is_running": false,
                      "invoice": null,
                      "external_reference": null,
                      "billable": true,
                      "budgeted": true,
                      "billable_rate": 100,
                      "cost_rate": 50
                    },
                    {
                      "id": 636708723,
                      "spent_date": "2017-03-01",
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      },
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1"
                      },
                      "task": {
                        "id": 8083366,
                        "name": "Programming"
                      },
                      "user_assignment": {
                        "id": 125068554,
                        "is_project_manager": true,
                        "is_active": true,
                        "budget": null,
                        "created_at": "2017-06-26T22:32:52Z",
                        "updated_at": "2017-06-26T22:32:52Z",
                        "hourly_rate": 100
                      },
                      "task_assignment": {
                        "id": 155505014,
                        "billable": true,
                        "is_active": true,
                        "created_at": "2017-06-26T21:52:18Z",
                        "updated_at": "2017-06-26T21:52:18Z",
                        "hourly_rate": 100,
                        "budget": null
                      },
                      "hours": 1.35,
                      "hours_without_timer": 1.35,
                      "rounded_hours": 1.5,
                      "notes": "Importing products",
                      "created_at": "2017-06-27T15:49:28Z",
                      "updated_at": "2017-06-27T16:47:14Z",
                      "is_locked": true,
                      "locked_reason": "Item Invoiced and Approved and Locked for this Time Period",
                      "is_closed": true,
                      "is_billed": true,
                      "timer_started_at": null,
                      "started_time": "1:00pm",
                      "ended_time": "2:00pm",
                      "is_running": false,
                      "invoice": {
                        "id": 13150403,
                        "number": "1001"
                      },
                      "external_reference": null,
                      "billable": true,
                      "budgeted": true,
                      "billable_rate": 100,
                      "cost_rate": 50
                    },
                    {
                      "id": 636708574,
                      "spent_date": "2017-03-01",
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      },
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1"
                      },
                      "task": {
                        "id": 8083369,
                        "name": "Research"
                      },
                      "user_assignment": {
                        "id": 125068554,
                        "is_project_manager": true,
                        "is_active": true,
                        "budget": null,
                        "created_at": "2017-06-26T22:32:52Z",
                        "updated_at": "2017-06-26T22:32:52Z",
                        "hourly_rate": 100
                      },
                      "task_assignment": {
                        "id": 155505016,
                        "billable": false,
                        "is_active": true,
                        "created_at": "2017-06-26T21:52:18Z",
                        "updated_at": "2017-06-26T21:54:06Z",
                        "hourly_rate": 100,
                        "budget": null
                      },
                      "hours": 1,
                      "hours_without_timer": 1,
                      "rounded_hours": 1,
                      "notes": "Evaluating 3rd party libraries",
                      "created_at": "2017-06-27T15:49:17Z",
                      "updated_at": "2017-06-27T16:47:14Z",
                      "is_locked": true,
                      "locked_reason": "Item Approved and Locked for this Time Period",
                      "is_closed": true,
                      "is_billed": false,
                      "timer_started_at": null,
                      "started_time": "11:00am",
                      "ended_time": "12:00pm",
                      "is_running": false,
                      "invoice": null,
                      "external_reference": null,
                      "billable": false,
                      "budgeted": true,
                      "billable_rate": null,
                      "cost_rate": 50
                    },
                    {
                      "id": 636707831,
                      "spent_date": "2017-03-01",
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      },
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1"
                      },
                      "task": {
                        "id": 8083368,
                        "name": "Project Management"
                      },
                      "user_assignment": {
                        "id": 125068554,
                        "is_project_manager": true,
                        "is_active": true,
                        "budget": null,
                        "created_at": "2017-06-26T22:32:52Z",
                        "updated_at": "2017-06-26T22:32:52Z",
                        "hourly_rate": 100
                      },
                      "task_assignment": {
                        "id": 155505015,
                        "billable": true,
                        "is_active": true,
                        "created_at": "2017-06-26T21:52:18Z",
                        "updated_at": "2017-06-26T21:52:18Z",
                        "hourly_rate": 100,
                        "budget": null
                      },
                      "hours": 2,
                      "hours_without_timer": 2,
                      "rounded_hours": 2,
                      "notes": "Planning meetings",
                      "created_at": "2017-06-27T15:48:24Z",
                      "updated_at": "2017-06-27T16:47:14Z",
                      "is_locked": true,
                      "locked_reason": "Item Invoiced and Approved and Locked for this Time Period",
                      "is_closed": true,
                      "is_billed": true,
                      "timer_started_at": null,
                      "started_time": "9:00am",
                      "ended_time": "11:00am",
                      "is_running": false,
                      "invoice": {
                        "id": 13150403,
                        "number": "1001"
                      },
                      "external_reference": null,
                      "billable": true,
                      "budgeted": true,
                      "billable_rate": 100,
                      "cost_rate": 50
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 4,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/time_entries?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/time_entries?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "user_id",
            "description": "Only return time entries belonging to the user with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "client_id",
            "description": "Only return time entries belonging to the client with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "project_id",
            "description": "Only return time entries belonging to the project with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "task_id",
            "description": "Only return time entries belonging to the task with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "external_reference_id",
            "description": "Only return time entries with the given external_reference ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "is_billed",
            "description": "Pass true to only return time entries that have been invoiced and false to return time entries that have not been invoiced.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "is_running",
            "description": "Pass true to only return running time entries and false to return non-running time entries.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return time entries that have been updated since the given date and time. Use the ISO 8601 Format.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "from",
            "description": "Only return time entries with a spent_date on or after the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "to",
            "description": "Only return time entries with a spent_date on or before the given date.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a time entry",
        "operationId": "createTimeEntry",
        "description": "Creates a new time entry object. Returns a time entry object and a 201 Created response code if the call succeeded.\n\nYou should only use this method to create time entries when your account is configured to track time via duration. You can verify this by visiting the Settings page in your Harvest account or by checking if wants_timestamp_timers is false in the Company API.",
        "externalDocs": {
          "description": "Create a time entry via duration",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#create-a-time-entry-via-duration"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a time entry",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeEntry"
                },
                "example": {
                  "id": 636718192,
                  "spent_date": "2017-03-21",
                  "user": {
                    "id": 1782959,
                    "name": "Kim Allen"
                  },
                  "client": {
                    "id": 5735774,
                    "name": "ABC Corp"
                  },
                  "project": {
                    "id": 14307913,
                    "name": "Marketing Website"
                  },
                  "task": {
                    "id": 8083365,
                    "name": "Graphic Design"
                  },
                  "user_assignment": {
                    "id": 125068553,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-06-26T22:32:52Z",
                    "updated_at": "2017-06-26T22:32:52Z",
                    "hourly_rate": 100
                  },
                  "task_assignment": {
                    "id": 155502709,
                    "billable": true,
                    "is_active": true,
                    "created_at": "2017-06-26T21:36:23Z",
                    "updated_at": "2017-06-26T21:36:23Z",
                    "hourly_rate": 100,
                    "budget": null
                  },
                  "hours": 1,
                  "rounded_hours": 1,
                  "notes": null,
                  "created_at": "2017-06-27T16:01:23Z",
                  "updated_at": "2017-06-27T16:01:23Z",
                  "is_locked": false,
                  "locked_reason": null,
                  "is_closed": false,
                  "is_billed": false,
                  "timer_started_at": null,
                  "started_time": null,
                  "ended_time": null,
                  "is_running": false,
                  "invoice": null,
                  "external_reference": null,
                  "billable": true,
                  "budgeted": true,
                  "billable_rate": 100,
                  "cost_rate": 50
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "user_id": {
                    "type": "integer",
                    "description": "The ID of the user to associate with the time entry. Defaults to the currently authenticated user’s ID.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "project_id": {
                    "type": "integer",
                    "description": "The ID of the project to associate with the time entry.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "task_id": {
                    "type": "integer",
                    "description": "The ID of the task to associate with the time entry.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "spent_date": {
                    "type": "string",
                    "description": "The ISO 8601 formatted date the time entry was spent.",
                    "nullable": true,
                    "format": "date"
                  },
                  "started_time": {
                    "type": "string",
                    "description": "The time the entry started. Defaults to the current time. Example: “8:00am”.",
                    "nullable": true
                  },
                  "ended_time": {
                    "type": "string",
                    "description": "The time the entry ended. If provided, is_running will be set to false. If not provided, is_running will be set to true.",
                    "nullable": true
                  },
                  "notes": {
                    "type": "string",
                    "description": "Any notes to be associated with the time entry.",
                    "nullable": true
                  },
                  "external_reference": {
                    "type": "object",
                    "description": "An object containing the id, group_id, account_id, and permalink of the external reference.",
                    "nullable": true,
                    "properties": {
                      "id": {
                        "type": "string",
                        "nullable": true
                      },
                      "group_id": {
                        "type": "string",
                        "nullable": true
                      },
                      "account_id": {
                        "type": "string",
                        "nullable": true
                      },
                      "permalink": {
                        "type": "string",
                        "nullable": true
                      }
                    }
                  },
                  "hours": {
                    "type": "number",
                    "description": "The current amount of time tracked. If provided, the time entry will be created with the specified hours and is_running will be set to false. If not provided, hours will be set to 0.0 and is_running will be set to true.",
                    "nullable": true,
                    "format": "float"
                  }
                },
                "required": [
                  "project_id",
                  "task_id",
                  "spent_date"
                ]
              }
            }
          }
        }
      }
    },
    "/time_entries/{timeEntryId}": {
      "delete": {
        "summary": "Delete a time entry",
        "operationId": "deleteTimeEntry",
        "description": "Delete a time entry. Deleting a time entry is only possible if it’s not closed and the associated project and task haven’t been archived.  However, Admins can delete closed entries. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a time entry",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#delete-a-time-entry"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a time entry"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "timeEntryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a time entry",
        "operationId": "retrieveTimeEntry",
        "description": "Retrieves the time entry with the given ID. Returns a time entry object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a time entry",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#retrieve-a-time-entry"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a time entry",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeEntry"
                },
                "example": {
                  "id": 636708723,
                  "spent_date": "2017-03-01",
                  "user": {
                    "id": 1782959,
                    "name": "Kim Allen"
                  },
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries"
                  },
                  "project": {
                    "id": 14308069,
                    "name": "Online Store - Phase 1"
                  },
                  "task": {
                    "id": 8083366,
                    "name": "Programming"
                  },
                  "user_assignment": {
                    "id": 125068554,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-06-26T22:32:52Z",
                    "updated_at": "2017-06-26T22:32:52Z",
                    "hourly_rate": 100
                  },
                  "task_assignment": {
                    "id": 155505014,
                    "billable": true,
                    "is_active": true,
                    "created_at": "2017-06-26T21:52:18Z",
                    "updated_at": "2017-06-26T21:52:18Z",
                    "hourly_rate": 100,
                    "budget": null
                  },
                  "hours": 1,
                  "hours_without_timer": 1,
                  "rounded_hours": 1,
                  "notes": "Importing products",
                  "created_at": "2017-06-27T15:49:28Z",
                  "updated_at": "2017-06-27T16:47:14Z",
                  "is_locked": true,
                  "locked_reason": "Item Invoiced and Approved and Locked for this Time Period",
                  "is_closed": true,
                  "is_billed": true,
                  "timer_started_at": null,
                  "started_time": "1:00pm",
                  "ended_time": "2:00pm",
                  "is_running": false,
                  "invoice": {
                    "id": 13150403,
                    "number": "1001"
                  },
                  "external_reference": null,
                  "billable": true,
                  "budgeted": true,
                  "billable_rate": 100,
                  "cost_rate": 50
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "timeEntryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a time entry",
        "operationId": "updateTimeEntry",
        "description": "Updates the specific time entry by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a time entry object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a time entry",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#update-a-time-entry"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a time entry",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeEntry"
                },
                "example": {
                  "id": 636718192,
                  "spent_date": "2017-03-21",
                  "user": {
                    "id": 1782959,
                    "name": "Kim Allen"
                  },
                  "client": {
                    "id": 5735774,
                    "name": "ABC Corp"
                  },
                  "project": {
                    "id": 14307913,
                    "name": "Marketing Website"
                  },
                  "task": {
                    "id": 8083365,
                    "name": "Graphic Design"
                  },
                  "user_assignment": {
                    "id": 125068553,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-06-26T22:32:52Z",
                    "updated_at": "2017-06-26T22:32:52Z",
                    "hourly_rate": 100
                  },
                  "task_assignment": {
                    "id": 155502709,
                    "billable": true,
                    "is_active": true,
                    "created_at": "2017-06-26T21:36:23Z",
                    "updated_at": "2017-06-26T21:36:23Z",
                    "hourly_rate": 100,
                    "budget": null
                  },
                  "hours": 1,
                  "hours_without_timer": 1,
                  "rounded_hours": 1,
                  "notes": "Updated notes",
                  "created_at": "2017-06-27T16:01:23Z",
                  "updated_at": "2017-06-27T16:02:40Z",
                  "is_locked": false,
                  "locked_reason": null,
                  "is_closed": false,
                  "is_billed": false,
                  "timer_started_at": null,
                  "started_time": null,
                  "ended_time": null,
                  "is_running": false,
                  "invoice": null,
                  "external_reference": null,
                  "billable": true,
                  "budgeted": true,
                  "billable_rate": 100,
                  "cost_rate": 50
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "timeEntryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "integer",
                    "description": "The ID of the project to associate with the time entry.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "task_id": {
                    "type": "integer",
                    "description": "The ID of the task to associate with the time entry.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "spent_date": {
                    "type": "string",
                    "description": "The ISO 8601 formatted date the time entry was spent.",
                    "nullable": true,
                    "format": "date"
                  },
                  "started_time": {
                    "type": "string",
                    "description": "The time the entry started. Defaults to the current time. Example: “8:00am”.",
                    "nullable": true
                  },
                  "ended_time": {
                    "type": "string",
                    "description": "The time the entry ended.",
                    "nullable": true
                  },
                  "hours": {
                    "type": "number",
                    "description": "The current amount of time tracked.",
                    "nullable": true,
                    "format": "float"
                  },
                  "notes": {
                    "type": "string",
                    "description": "Any notes to be associated with the time entry.",
                    "nullable": true
                  },
                  "external_reference": {
                    "type": "object",
                    "description": "An object containing the id, group_id, account_id, and permalink of the external reference.",
                    "nullable": true,
                    "properties": {
                      "id": {
                        "type": "string",
                        "nullable": true
                      },
                      "group_id": {
                        "type": "string",
                        "nullable": true
                      },
                      "account_id": {
                        "type": "string",
                        "nullable": true
                      },
                      "permalink": {
                        "type": "string",
                        "nullable": true
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/time_entries/{timeEntryId}/external_reference": {
      "delete": {
        "summary": "Delete a time entry’s external reference",
        "operationId": "deleteTimeEntryExternalReference",
        "description": "Delete a time entry’s external reference. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a time entry’s external reference",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#delete-a-time-entrys-external-reference"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a time entry’s external reference"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "timeEntryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/time_entries/{timeEntryId}/restart": {
      "patch": {
        "summary": "Restart a stopped time entry",
        "operationId": "restartStoppedTimeEntry",
        "description": "Restarting a time entry is only possible if it isn’t currently running. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Restart a stopped time entry",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#restart-a-stopped-time-entry"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Restart a stopped time entry",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeEntry"
                },
                "example": {
                  "id": 662204379,
                  "spent_date": "2017-03-21",
                  "user": {
                    "id": 1795925,
                    "name": "Jane Smith"
                  },
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries"
                  },
                  "project": {
                    "id": 14808188,
                    "name": "Task Force"
                  },
                  "task": {
                    "id": 8083366,
                    "name": "Programming"
                  },
                  "user_assignment": {
                    "id": 130403296,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-08-22T17:36:54Z",
                    "updated_at": "2017-08-22T17:36:54Z",
                    "hourly_rate": 100
                  },
                  "task_assignment": {
                    "id": 160726645,
                    "billable": true,
                    "is_active": true,
                    "created_at": "2017-08-22T17:36:54Z",
                    "updated_at": "2017-08-22T17:36:54Z",
                    "hourly_rate": 100,
                    "budget": null
                  },
                  "hours": 0,
                  "hours_without_timer": 0,
                  "rounded_hours": 0,
                  "notes": null,
                  "created_at": "2017-08-22T17:40:24Z",
                  "updated_at": "2017-08-22T17:40:24Z",
                  "is_locked": false,
                  "locked_reason": null,
                  "is_closed": false,
                  "is_billed": false,
                  "timer_started_at": "2017-08-22T17:40:24Z",
                  "started_time": "11:40am",
                  "ended_time": null,
                  "is_running": true,
                  "invoice": null,
                  "external_reference": null,
                  "billable": true,
                  "budgeted": false,
                  "billable_rate": 100,
                  "cost_rate": 75
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "timeEntryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/time_entries/{timeEntryId}/stop": {
      "patch": {
        "summary": "Stop a running time entry",
        "operationId": "stopRunningTimeEntry",
        "description": "Stopping a time entry is only possible if it’s currently running. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Stop a running time entry",
          "url": "https://help.getharvest.com/api-v2/timesheets-api/timesheets/time-entries/#stop-a-running-time-entry"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Stop a running time entry",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TimeEntry"
                },
                "example": {
                  "id": 662202797,
                  "spent_date": "2017-03-21",
                  "user": {
                    "id": 1795925,
                    "name": "Jane Smith"
                  },
                  "client": {
                    "id": 5735776,
                    "name": "123 Industries"
                  },
                  "project": {
                    "id": 14808188,
                    "name": "Task Force"
                  },
                  "task": {
                    "id": 8083366,
                    "name": "Programming"
                  },
                  "user_assignment": {
                    "id": 130403296,
                    "is_project_manager": true,
                    "is_active": true,
                    "budget": null,
                    "created_at": "2017-08-22T17:36:54Z",
                    "updated_at": "2017-08-22T17:36:54Z",
                    "hourly_rate": 100
                  },
                  "task_assignment": {
                    "id": 160726645,
                    "billable": true,
                    "is_active": true,
                    "created_at": "2017-08-22T17:36:54Z",
                    "updated_at": "2017-08-22T17:36:54Z",
                    "hourly_rate": 100,
                    "budget": null
                  },
                  "hours": 0.02,
                  "hours_without_timer": 0.02,
                  "rounded_hours": 0.25,
                  "notes": null,
                  "created_at": "2017-08-22T17:37:13Z",
                  "updated_at": "2017-08-22T17:38:31Z",
                  "is_locked": false,
                  "locked_reason": null,
                  "is_closed": false,
                  "is_billed": false,
                  "timer_started_at": null,
                  "started_time": "11:37am",
                  "ended_time": "11:38am",
                  "is_running": false,
                  "invoice": null,
                  "external_reference": null,
                  "billable": true,
                  "budgeted": false,
                  "billable_rate": 100,
                  "cost_rate": 75
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "timeEntryId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/user_assignments": {
      "get": {
        "summary": "List all user assignments",
        "operationId": "listUserAssignments",
        "description": "Returns a list of your projects user assignments, active and archived. The user assignments are returned sorted by creation date, with the most recently created user assignments appearing first.\n\nThe response contains an object with a user_assignments property that contains an array of up to per_page user assignments. Each entry in the array is a separate user assignment object. If no more user assignments are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your user assignments.",
        "externalDocs": {
          "description": "List all user assignments",
          "url": "https://help.getharvest.com/api-v2/projects-api/projects/user-assignments/#list-all-user-assignments"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all user assignments",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UserAssignments"
                },
                "example": {
                  "user_assignments": [
                    {
                      "id": 130403297,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": false,
                      "budget": null,
                      "created_at": "2017-08-22T17:36:54Z",
                      "updated_at": "2017-08-22T17:36:54Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14808188,
                        "name": "Task Force",
                        "code": "TF"
                      },
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      }
                    },
                    {
                      "id": 130403296,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": true,
                      "budget": null,
                      "created_at": "2017-08-22T17:36:54Z",
                      "updated_at": "2017-08-22T17:36:54Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14808188,
                        "name": "Task Force",
                        "code": "TF"
                      },
                      "user": {
                        "id": 1795925,
                        "name": "Jason Dew"
                      }
                    },
                    {
                      "id": 125068554,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": true,
                      "budget": null,
                      "created_at": "2017-06-26T22:32:52Z",
                      "updated_at": "2017-06-26T22:32:52Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      }
                    },
                    {
                      "id": 125068553,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": true,
                      "budget": null,
                      "created_at": "2017-06-26T22:32:52Z",
                      "updated_at": "2017-06-26T22:32:52Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "user": {
                        "id": 1782959,
                        "name": "Kim Allen"
                      }
                    },
                    {
                      "id": 125066109,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": false,
                      "budget": null,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "user": {
                        "id": 1782884,
                        "name": "Jeremy Israelsen"
                      }
                    },
                    {
                      "id": 125063975,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": true,
                      "budget": null,
                      "created_at": "2017-06-26T21:36:23Z",
                      "updated_at": "2017-06-26T21:36:23Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "user": {
                        "id": 1782884,
                        "name": "Jeremy Israelsen"
                      }
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 6,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/user_assignments?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/user_assignments?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "user_id",
            "description": "Only return user assignments belonging to the user with the given ID.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "is_active",
            "description": "Pass true to only return active user assignments and false to return inactive user assignments.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return user assignments that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/users": {
      "get": {
        "summary": "List all users",
        "operationId": "listUsers",
        "description": "Returns a list of your users. The users are returned sorted by creation date, with the most recently created users appearing first.\n\nThe response contains an object with a users property that contains an array of up to per_page users. Each entry in the array is a separate user object. If no more users are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your users.",
        "externalDocs": {
          "description": "List all users",
          "url": "https://help.getharvest.com/api-v2/users-api/users/users/#list-all-users"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all users",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Users"
                },
                "example": {
                  "users": [
                    {
                      "id": 3230547,
                      "first_name": "Jim",
                      "last_name": "Allen",
                      "email": "jimallen@example.com",
                      "telephone": "",
                      "timezone": "Mountain Time (US & Canada)",
                      "has_access_to_all_future_projects": false,
                      "is_contractor": false,
                      "is_active": true,
                      "created_at": "2020-05-01T22:34:41Z",
                      "updated_at": "2020-05-01T22:34:52Z",
                      "weekly_capacity": 126000,
                      "default_hourly_rate": 100,
                      "cost_rate": 50,
                      "roles": [
                        "Developer"
                      ],
                      "access_roles": [
                        "member"
                      ],
                      "avatar_url": "https://cache.harvestapp.com/assets/profile_images/abraj_albait_towers.png?1498516481"
                    },
                    {
                      "id": 1782959,
                      "first_name": "Kim",
                      "last_name": "Allen",
                      "email": "kimallen@example.com",
                      "telephone": "",
                      "timezone": "Eastern Time (US & Canada)",
                      "has_access_to_all_future_projects": true,
                      "is_contractor": false,
                      "is_active": true,
                      "created_at": "2020-05-01T22:15:45Z",
                      "updated_at": "2020-05-01T22:32:52Z",
                      "weekly_capacity": 126000,
                      "default_hourly_rate": 100,
                      "cost_rate": 50,
                      "roles": [
                        "Designer"
                      ],
                      "access_roles": [
                        "member"
                      ],
                      "avatar_url": "https://cache.harvestapp.com/assets/profile_images/cornell_clock_tower.png?1498515345"
                    },
                    {
                      "id": 1782884,
                      "first_name": "Bob",
                      "last_name": "Powell",
                      "email": "bobpowell@example.com",
                      "telephone": "",
                      "timezone": "Mountain Time (US & Canada)",
                      "has_access_to_all_future_projects": false,
                      "is_contractor": false,
                      "is_active": true,
                      "created_at": "2020-05-01T20:41:00Z",
                      "updated_at": "2020-05-01T20:42:25Z",
                      "weekly_capacity": 126000,
                      "default_hourly_rate": 100,
                      "cost_rate": 75,
                      "roles": [
                        "Founder",
                        "CEO"
                      ],
                      "access_roles": [
                        "administrator"
                      ],
                      "avatar_url": "https://cache.harvestapp.com/assets/profile_images/allen_bradley_clock_tower.png?1498509661"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 3,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/users?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/users?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "is_active",
            "description": "Pass true to only return active users and false to return inactive users.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "boolean"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return users that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a user",
        "operationId": "createUser",
        "description": "Creates a new user object and sends an invitation email to the address specified in the email parameter. Returns a user object and a 201 Created response code if the call succeeded.",
        "externalDocs": {
          "description": "Create a user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/users/#create-a-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/User"
                },
                "example": {
                  "id": 3,
                  "first_name": "George",
                  "last_name": "Frank",
                  "email": "george@example.com",
                  "telephone": "",
                  "timezone": "Eastern Time (US & Canada)",
                  "has_access_to_all_future_projects": false,
                  "is_contractor": false,
                  "is_active": true,
                  "weekly_capacity": 126000,
                  "default_hourly_rate": 0,
                  "cost_rate": 0,
                  "roles": [],
                  "access_roles": [
                    "manager",
                    "project_creator",
                    "time_and_expenses_manager"
                  ],
                  "avatar_url": "https://{ACCOUNT_SUBDOMAIN}.harvestapp.com/assets/profile_images/big_ben.png?1485372046",
                  "created_at": "2020-01-25T19:20:46Z",
                  "updated_at": "2020-01-25T19:20:57Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "first_name": {
                    "type": "string",
                    "description": "The first name of the user.",
                    "nullable": true
                  },
                  "last_name": {
                    "type": "string",
                    "description": "The last name of the user.",
                    "nullable": true
                  },
                  "email": {
                    "type": "string",
                    "description": "The email address of the user.",
                    "nullable": true,
                    "format": "email"
                  },
                  "timezone": {
                    "type": "string",
                    "description": "The user’s timezone. Defaults to the company’s timezone. See a list of supported time zones.",
                    "nullable": true
                  },
                  "has_access_to_all_future_projects": {
                    "type": "boolean",
                    "description": "Whether the user should be automatically added to future projects. Defaults to false.",
                    "nullable": true
                  },
                  "is_contractor": {
                    "type": "boolean",
                    "description": "Whether the user is a contractor or an employee. Defaults to false.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the user is active or archived. Defaults to true.",
                    "nullable": true
                  },
                  "weekly_capacity": {
                    "type": "integer",
                    "description": "The number of hours per week this person is available to work in seconds. Defaults to 126000 seconds (35 hours).",
                    "nullable": true,
                    "format": "int32"
                  },
                  "default_hourly_rate": {
                    "type": "number",
                    "description": "The billable rate to use for this user when they are added to a project. Defaults to 0.",
                    "nullable": true,
                    "format": "float"
                  },
                  "cost_rate": {
                    "type": "number",
                    "description": "The cost rate to use for this user when calculating a project’s costs vs billable amount. Defaults to 0.",
                    "nullable": true,
                    "format": "float"
                  },
                  "roles": {
                    "type": "array",
                    "description": "Descriptive names of the business roles assigned to this person. They can be used for filtering reports, and have no effect in their permissions in Harvest.",
                    "nullable": true,
                    "items": {
                      "type": "string"
                    }
                  },
                  "access_roles": {
                    "type": "array",
                    "description": "Access role(s) that determine the user’s permissions in Harvest. Possible values: administrator, manager or member. Users with the manager role can additionally be granted one or more of these roles: project_creator, billable_rates_manager, managed_projects_invoice_drafter, managed_projects_invoice_manager, client_and_task_manager, time_and_expenses_manager, estimates_manager.",
                    "nullable": true,
                    "items": {
                      "type": "string"
                    }
                  }
                },
                "required": [
                  "first_name",
                  "last_name",
                  "email"
                ]
              }
            }
          }
        }
      }
    },
    "/users/me": {
      "get": {
        "summary": "Retrieve the currently authenticated user",
        "operationId": "retrieveTheCurrentlyAuthenticatedUser",
        "description": "Retrieves the currently authenticated user. Returns a user object and a 200 OK response code.",
        "externalDocs": {
          "description": "Retrieve the currently authenticated user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/users/#retrieve-the-currently-authenticated-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve the currently authenticated user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/User"
                },
                "example": {
                  "id": 1782884,
                  "first_name": "Bob",
                  "last_name": "Powell",
                  "email": "bobpowell@example.com",
                  "telephone": "",
                  "timezone": "Mountain Time (US & Canada)",
                  "has_access_to_all_future_projects": false,
                  "is_contractor": false,
                  "is_active": true,
                  "created_at": "2020-05-01T20:41:00Z",
                  "updated_at": "2020-05-01T20:42:25Z",
                  "weekly_capacity": 126000,
                  "default_hourly_rate": 100,
                  "cost_rate": 75,
                  "roles": [
                    "Founder",
                    "CEO"
                  ],
                  "access_roles": [
                    "administrator"
                  ],
                  "avatar_url": "https://cache.harvestapp.com/assets/profile_images/allen_bradley_clock_tower.png?1498509661"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/users/me/project_assignments": {
      "get": {
        "summary": "List active project assignments for the currently authenticated user",
        "operationId": "listActiveProjectAssignmentsForTheCurrentlyAuthenticatedUser",
        "description": "Returns a list of your active project assignments for the currently authenticated user. The project assignments are returned sorted by creation date, with the most recently created project assignments appearing first.\n\nThe response contains an object with a project_assignments property that contains an array of up to per_page project assignments. Each entry in the array is a separate project assignment object. If no more project assignments are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your project assignments.",
        "externalDocs": {
          "description": "List active project assignments for the currently authenticated user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/project-assignments/#list-active-project-assignments-for-the-currently-authenticated-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List active project assignments for the currently authenticated user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ProjectAssignments"
                },
                "example": {
                  "project_assignments": [
                    {
                      "id": 125066109,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": true,
                      "budget": null,
                      "created_at": "2017-06-26T21:52:18Z",
                      "updated_at": "2017-06-26T21:52:18Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "task_assignments": [
                        {
                          "id": 155505013,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:52:18Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083365,
                            "name": "Graphic Design"
                          }
                        },
                        {
                          "id": 155505014,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:52:18Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083366,
                            "name": "Programming"
                          }
                        },
                        {
                          "id": 155505015,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:52:18Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083368,
                            "name": "Project Management"
                          }
                        },
                        {
                          "id": 155505016,
                          "billable": false,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:54:06Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083369,
                            "name": "Research"
                          }
                        }
                      ]
                    },
                    {
                      "id": 125063975,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": false,
                      "budget": null,
                      "created_at": "2017-06-26T21:36:23Z",
                      "updated_at": "2017-06-26T21:36:23Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "client": {
                        "id": 5735774,
                        "name": "ABC Corp"
                      },
                      "task_assignments": [
                        {
                          "id": 155502709,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:36:23Z",
                          "updated_at": "2017-06-26T21:36:23Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083365,
                            "name": "Graphic Design"
                          }
                        },
                        {
                          "id": 155502710,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:36:23Z",
                          "updated_at": "2017-06-26T21:36:23Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083366,
                            "name": "Programming"
                          }
                        },
                        {
                          "id": 155502711,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:36:23Z",
                          "updated_at": "2017-06-26T21:36:23Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083368,
                            "name": "Project Management"
                          }
                        },
                        {
                          "id": 155505153,
                          "billable": false,
                          "is_active": true,
                          "created_at": "2017-06-26T21:53:20Z",
                          "updated_at": "2017-06-26T21:54:31Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083369,
                            "name": "Research"
                          }
                        }
                      ]
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/users/1782884/project_assignments?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/users/1782884/project_assignments?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "page",
            "description": "The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/users/{userId}": {
      "delete": {
        "summary": "Delete a user",
        "operationId": "deleteUser",
        "description": "Delete a user. Deleting a user is only possible if they have no time entries or expenses associated with them. Returns a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Delete a user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/users/#delete-a-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Delete a user"
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "get": {
        "summary": "Retrieve a user",
        "operationId": "retrieveUser",
        "description": "Retrieves the user with the given ID. Returns a user object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/users/#retrieve-a-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/User"
                },
                "example": {
                  "id": 3230547,
                  "first_name": "Jim",
                  "last_name": "Allen",
                  "email": "jimallen@example.com",
                  "telephone": "",
                  "timezone": "Mountain Time (US & Canada)",
                  "has_access_to_all_future_projects": false,
                  "is_contractor": false,
                  "is_active": true,
                  "created_at": "2020-05-01T22:34:41Z",
                  "updated_at": "2020-05-01T22:34:52Z",
                  "weekly_capacity": 126000,
                  "default_hourly_rate": 100,
                  "cost_rate": 50,
                  "roles": [
                    "Developer"
                  ],
                  "access_roles": [
                    "member"
                  ],
                  "avatar_url": "https://cache.harvestapp.com/assets/profile_images/abraj_albait_towers.png?1498516481"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a user",
        "operationId": "updateUser",
        "description": "Updates the specific user by setting the values of the parameters passed. Any parameters not provided will be left unchanged. Returns a user object and a 200 OK response code if the call succeeded.",
        "externalDocs": {
          "description": "Update a user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/users/#update-a-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/User"
                },
                "example": {
                  "id": 3237198,
                  "first_name": "Gary",
                  "last_name": "Brookes",
                  "email": "gary@example.com",
                  "telephone": "",
                  "timezone": "Eastern Time (US & Canada)",
                  "has_access_to_all_future_projects": true,
                  "is_contractor": false,
                  "is_active": true,
                  "weekly_capacity": 126000,
                  "default_hourly_rate": 120,
                  "cost_rate": 50,
                  "roles": [
                    "Product Team"
                  ],
                  "access_roles": [
                    "manager",
                    "time_and_expenses_manager",
                    "billable_rates_manager"
                  ],
                  "avatar_url": "https://{ACCOUNT_SUBDOMAIN}.harvestapp.com/assets/profile_images/big_ben.png?1485372046",
                  "created_at": "2018-01-01T19:20:46Z",
                  "updated_at": "2019-01-25T19:20:57Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "first_name": {
                    "type": "string",
                    "description": "The first name of the user. Can’t be updated if the user is inactive.",
                    "nullable": true
                  },
                  "last_name": {
                    "type": "string",
                    "description": "The last name of the user. Can’t be updated if the user is inactive.",
                    "nullable": true
                  },
                  "email": {
                    "type": "string",
                    "description": "The email address of the user. Can’t be updated if the user is inactive.",
                    "nullable": true,
                    "format": "email"
                  },
                  "timezone": {
                    "type": "string",
                    "description": "The user’s timezone. Defaults to the company’s timezone. See a list of supported time zones.",
                    "nullable": true
                  },
                  "has_access_to_all_future_projects": {
                    "type": "boolean",
                    "description": "Whether the user should be automatically added to future projects.",
                    "nullable": true
                  },
                  "is_contractor": {
                    "type": "boolean",
                    "description": "Whether the user is a contractor or an employee.",
                    "nullable": true
                  },
                  "is_active": {
                    "type": "boolean",
                    "description": "Whether the user is active or archived.",
                    "nullable": true
                  },
                  "weekly_capacity": {
                    "type": "integer",
                    "description": "The number of hours per week this person is available to work in seconds.",
                    "nullable": true,
                    "format": "int32"
                  },
                  "default_hourly_rate": {
                    "type": "number",
                    "description": "The billable rate to use for this user when they are added to a project.",
                    "nullable": true,
                    "format": "float"
                  },
                  "cost_rate": {
                    "type": "number",
                    "description": "The cost rate to use for this user when calculating a project’s costs vs billable amount.",
                    "nullable": true,
                    "format": "float"
                  },
                  "roles": {
                    "type": "array",
                    "description": "Descriptive names of the business roles assigned to this person. They can be used for filtering reports, and have no effect in their permissions in Harvest.",
                    "nullable": true,
                    "items": {
                      "type": "string"
                    }
                  },
                  "access_roles": {
                    "type": "array",
                    "description": "Access role(s) that determine the user’s permissions in Harvest. Possible values: administrator, manager or member. Users with the manager role can additionally be granted one or more of these roles: project_creator, billable_rates_manager, managed_projects_invoice_drafter, managed_projects_invoice_manager, client_and_task_manager, time_and_expenses_manager, estimates_manager.",
                    "nullable": true,
                    "items": {
                      "type": "string"
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/users/{userId}/billable_rates": {
      "get": {
        "summary": "List all billable rates for a specific user",
        "operationId": "listBillableRatesForSpecificUser",
        "description": "Returns a list of billable rates for the user identified by USER_ID. The billable rates are returned sorted by start_date, with the oldest starting billable rates appearing first.\n\nThe response contains an object with a billable_rates property that contains an array of up to per_page billable rates. Each entry in the array is a separate billable rate object. If no more billable rates are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your billable rates.",
        "externalDocs": {
          "description": "List all billable rates for a specific user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/billable-rates/#list-all-billable-rates-for-a-specific-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all billable rates for a specific user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/BillableRates"
                },
                "example": {
                  "billable_rates": [
                    {
                      "id": 1836493,
                      "amount": 8.25,
                      "start_date": "2019-01-01",
                      "end_date": "2019-05-31",
                      "created_at": "2020-05-01T13:17:42Z",
                      "updated_at": "2020-05-01T13:17:50Z"
                    },
                    {
                      "id": 1836494,
                      "amount": 9.5,
                      "start_date": "2019-06-01",
                      "end_date": "2019-12-31",
                      "created_at": "2020-05-01T13:17:50Z",
                      "updated_at": "2020-05-01T13:18:02Z"
                    },
                    {
                      "id": 1836495,
                      "amount": 9.5,
                      "start_date": "2020-01-01",
                      "end_date": "2020-04-30",
                      "created_at": "2020-05-01T13:18:02Z",
                      "updated_at": "2020-05-01T13:18:10Z"
                    },
                    {
                      "id": 1836496,
                      "amount": 15,
                      "start_date": "2020-05-01",
                      "end_date": null,
                      "created_at": "2020-05-01T13:18:10Z",
                      "updated_at": "2020-05-01T13:18:10Z"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 4,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/users/3226125/billable_rates?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/users/3226125/billable_rates?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a billable rate",
        "operationId": "createBillableRate",
        "description": "Creates a new billable rate object. Returns a billable rate object and a 201 Created response code if the call succeeded.\n\n\n  Creating a billable rate with no start_date will replace a user’s existing rate(s).\n  Creating a billable rate with a start_date that is before a user’s existing rate(s) will replace those billable rates with the new one.\n",
        "externalDocs": {
          "description": "Create a billable rate",
          "url": "https://help.getharvest.com/api-v2/users-api/users/billable-rates/#create-a-billable-rate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a billable rate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/BillableRate"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "amount": {
                    "type": "number",
                    "description": "The amount of the billable rate.",
                    "nullable": true,
                    "format": "float"
                  },
                  "start_date": {
                    "type": "string",
                    "description": "The date the billable rate is effective. Cannot be a date in the future.",
                    "nullable": true,
                    "format": "date"
                  }
                },
                "required": [
                  "amount"
                ]
              }
            }
          }
        }
      }
    },
    "/users/{userId}/billable_rates/{billableRateId}": {
      "get": {
        "summary": "Retrieve a billable rate",
        "operationId": "retrieveBillableRate",
        "description": "Retrieves the billable rate with the given ID. Returns a billable rate object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a billable rate",
          "url": "https://help.getharvest.com/api-v2/users-api/users/billable-rates/#retrieve-a-billable-rate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a billable rate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/BillableRate"
                },
                "example": {
                  "id": 1836493,
                  "amount": 8.25,
                  "start_date": "2019-01-01",
                  "end_date": "2019-05-31",
                  "created_at": "2020-05-01T13:17:42Z",
                  "updated_at": "2020-05-01T13:17:50Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "billableRateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/users/{userId}/cost_rates": {
      "get": {
        "summary": "List all cost rates for a specific user",
        "operationId": "listCostRatesForSpecificUser",
        "description": "Returns a list of cost rates for the user identified by USER_ID. The cost rates are returned sorted by start_date, with the oldest starting cost rates appearing first.\n\nThe response contains an object with a cost_rates property that contains an array of up to per_page cost rates. Each entry in the array is a separate cost rate object. If no more cost rates are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your cost rates.",
        "externalDocs": {
          "description": "List all cost rates for a specific user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/cost-rates/#list-all-cost-rates-for-a-specific-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all cost rates for a specific user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostRates"
                },
                "example": {
                  "cost_rates": [
                    {
                      "id": 825301,
                      "amount": 9.25,
                      "start_date": "2019-01-01",
                      "end_date": "2019-05-31",
                      "created_at": "2020-05-01T13:19:09Z",
                      "updated_at": "2020-05-01T13:19:17Z"
                    },
                    {
                      "id": 825302,
                      "amount": 11,
                      "start_date": "2019-06-01",
                      "end_date": "2019-12-31",
                      "created_at": "2020-05-01T13:19:17Z",
                      "updated_at": "2020-05-01T13:19:24Z"
                    },
                    {
                      "id": 825303,
                      "amount": 12.5,
                      "start_date": "2020-01-01",
                      "end_date": "2020-04-30",
                      "created_at": "2020-05-01T13:19:24Z",
                      "updated_at": "2020-05-01T13:19:31Z"
                    },
                    {
                      "id": 825304,
                      "amount": 15.25,
                      "start_date": "2020-05-01",
                      "end_date": null,
                      "created_at": "2020-05-01T13:19:31Z",
                      "updated_at": "2020-05-01T13:19:31Z"
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 4,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/users/3226125/cost_rates?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/users/3226125/cost_rates?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "post": {
        "summary": "Create a cost rate",
        "operationId": "createCostRate",
        "description": "Creates a new cost rate object. Returns a cost rate object and a 201 Created response code if the call succeeded.\n\n\n  Creating a cost rate with no start_date will replace a user’s existing rate(s).\n  Creating a cost rate with a start_date that is before a user’s existing rate(s) will replace those cost rates with the new one.\n",
        "externalDocs": {
          "description": "Create a cost rate",
          "url": "https://help.getharvest.com/api-v2/users-api/users/cost-rates/#create-a-cost-rate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "201": {
            "description": "Create a cost rate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostRate"
                },
                "example": {
                  "id": 825305,
                  "amount": 13,
                  "start_date": "2020-04-05",
                  "end_date": null,
                  "created_at": "2020-05-01T13:23:27Z",
                  "updated_at": "2020-05-01T13:23:27Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "amount": {
                    "type": "number",
                    "description": "The amount of the cost rate.",
                    "nullable": true,
                    "format": "float"
                  },
                  "start_date": {
                    "type": "string",
                    "description": "The date the cost rate is effective. Cannot be a date in the future.",
                    "nullable": true,
                    "format": "date"
                  }
                },
                "required": [
                  "amount"
                ]
              }
            }
          }
        }
      }
    },
    "/users/{userId}/cost_rates/{costRateId}": {
      "get": {
        "summary": "Retrieve a cost rate",
        "operationId": "retrieveCostRate",
        "description": "Retrieves the cost rate with the given ID. Returns a cost rate object and a 200 OK response code if a valid identifier was provided.",
        "externalDocs": {
          "description": "Retrieve a cost rate",
          "url": "https://help.getharvest.com/api-v2/users-api/users/cost-rates/#retrieve-a-cost-rate"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Retrieve a cost rate",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CostRate"
                },
                "example": {
                  "id": 825301,
                  "amount": 9.25,
                  "start_date": "2019-01-01",
                  "end_date": "2019-05-31",
                  "created_at": "2020-05-01T13:19:09Z",
                  "updated_at": "2020-05-01T13:19:17Z"
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "costRateId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ]
      }
    },
    "/users/{userId}/project_assignments": {
      "get": {
        "summary": "List active project assignments",
        "operationId": "listActiveProjectAssignments",
        "description": "Returns a list of active project assignments for the user identified by USER_ID. The project assignments are returned sorted by creation date, with the most recently created project assignments appearing first.\n\nThe response contains an object with a project_assignments property that contains an array of up to per_page project assignments. Each entry in the array is a separate project assignment object. If no more project assignments are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your project assignments.",
        "externalDocs": {
          "description": "List active project assignments",
          "url": "https://help.getharvest.com/api-v2/users-api/users/project-assignments/#list-active-project-assignments"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List active project assignments",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ProjectAssignments"
                },
                "example": {
                  "project_assignments": [
                    {
                      "id": 125068554,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": true,
                      "budget": null,
                      "created_at": "2017-06-26T22:32:52Z",
                      "updated_at": "2017-06-26T22:32:52Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14308069,
                        "name": "Online Store - Phase 1",
                        "code": "OS1"
                      },
                      "client": {
                        "id": 5735776,
                        "name": "123 Industries"
                      },
                      "task_assignments": [
                        {
                          "id": 155505013,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:52:18Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083365,
                            "name": "Graphic Design"
                          }
                        },
                        {
                          "id": 155505014,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:52:18Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083366,
                            "name": "Programming"
                          }
                        },
                        {
                          "id": 155505015,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:52:18Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083368,
                            "name": "Project Management"
                          }
                        },
                        {
                          "id": 155505016,
                          "billable": false,
                          "is_active": true,
                          "created_at": "2017-06-26T21:52:18Z",
                          "updated_at": "2017-06-26T21:54:06Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083369,
                            "name": "Research"
                          }
                        }
                      ]
                    },
                    {
                      "id": 125068553,
                      "is_project_manager": true,
                      "is_active": true,
                      "use_default_rates": false,
                      "budget": null,
                      "created_at": "2017-06-26T22:32:52Z",
                      "updated_at": "2017-06-26T22:32:52Z",
                      "hourly_rate": 100,
                      "project": {
                        "id": 14307913,
                        "name": "Marketing Website",
                        "code": "MW"
                      },
                      "client": {
                        "id": 5735774,
                        "name": "ABC Corp"
                      },
                      "task_assignments": [
                        {
                          "id": 155502709,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:36:23Z",
                          "updated_at": "2017-06-26T21:36:23Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083365,
                            "name": "Graphic Design"
                          }
                        },
                        {
                          "id": 155502710,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:36:23Z",
                          "updated_at": "2017-06-26T21:36:23Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083366,
                            "name": "Programming"
                          }
                        },
                        {
                          "id": 155502711,
                          "billable": true,
                          "is_active": true,
                          "created_at": "2017-06-26T21:36:23Z",
                          "updated_at": "2017-06-26T21:36:23Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083368,
                            "name": "Project Management"
                          }
                        },
                        {
                          "id": 155505153,
                          "billable": false,
                          "is_active": true,
                          "created_at": "2017-06-26T21:53:20Z",
                          "updated_at": "2017-06-26T21:54:31Z",
                          "hourly_rate": 100,
                          "budget": null,
                          "task": {
                            "id": 8083369,
                            "name": "Research"
                          }
                        }
                      ]
                    }
                  ],
                  "per_page": 2000,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/users/1782959/project_assignments?page=1&per_page=2000",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/users/1782959/project_assignments?page=1&per_page=2000"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "updated_since",
            "description": "Only return project assignments that have been updated since the given date and time.",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 2000 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      }
    },
    "/users/{userId}/teammates": {
      "get": {
        "summary": "List all assigned teammates for a specific user",
        "operationId": "listAssignedTeammatesForSpecificUser",
        "description": "Returns a list of assigned teammates for the user identified by USER_ID. The USER_ID must belong to a user that is a Manager, if not, a 422 Unprocessable Entity status code will be returned.\n\nThe response contains an object with a teammates property that contains an array of up to per_page teammates. Each entry in the array is a separate teammate object. If no more teammates are available, the resulting array will be empty. Several additional pagination properties are included in the response to simplify paginating your teammates.",
        "externalDocs": {
          "description": "List all assigned teammates for a specific user",
          "url": "https://help.getharvest.com/api-v2/users-api/users/teammates/#list-all-assigned-teammates-for-a-specific-user"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "List all assigned teammates for a specific user",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Teammates"
                },
                "example": {
                  "teammates": [
                    {
                      "id": 3230547,
                      "first_name": "Jim",
                      "last_name": "Allen",
                      "email": "jimallen@example.com"
                    },
                    {
                      "id": 1782884,
                      "first_name": "Bob",
                      "last_name": "Powell",
                      "email": "bobpowell@example.com"
                    }
                  ],
                  "per_page": 100,
                  "total_pages": 1,
                  "total_entries": 2,
                  "next_page": null,
                  "previous_page": null,
                  "page": 1,
                  "links": {
                    "first": "https://api.harvestapp.com/v2/users/1782959/teammates?page=1&per_page=100",
                    "next": null,
                    "previous": null,
                    "last": "https://api.harvestapp.com/v2/users/1782959/teammates?page=1&per_page=100"
                  }
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "page",
            "description": "DEPRECATED The page number to use in pagination. For instance, if you make a list request and receive 100 records, your subsequent call can include page=2 to retrieve the next page of the list. (Default: 1)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            },
            "deprecated": true
          },
          {
            "name": "cursor",
            "description": "Pagination cursor",
            "required": false,
            "in": "query",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "per_page",
            "description": "The number of records to return per page. Can range between 1 and 2000. (Default: 2000)",
            "required": false,
            "in": "query",
            "schema": {
              "type": "integer"
            }
          }
        ]
      },
      "patch": {
        "summary": "Update a user’s assigned teammates",
        "operationId": "updateUserAssignedTeammates",
        "description": "Updates the the assigned teammates for a specific user. Returns list of assigned teammates and a 200 OK response code if the call succeeded. The USER_ID must belong to a user that is a Manager, if not, a 422 Unprocessable Entity status code will be returned.\n\nAdding teammates for the first time will add the people_manager access role to the Manager. Any IDs not included in the teammate_ids that are currently assigned will be unassigned from the Manager. Use an empty array to unassign all users. This will also remove the people_manager access role from the Manager.",
        "externalDocs": {
          "description": "Update a user’s assigned teammates",
          "url": "https://help.getharvest.com/api-v2/users-api/users/teammates/#update-a-users-assigned-teammates"
        },
        "security": [
          {
            "BearerAuth": [],
            "AccountAuth": []
          }
        ],
        "responses": {
          "200": {
            "description": "Update a user’s assigned teammates",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TeammatesPatchResponse"
                },
                "example": {
                  "teammates": [
                    {
                      "id": 3230547,
                      "first_name": "Jim",
                      "last_name": "Allen",
                      "email": "jimallen@example.com"
                    },
                    {
                      "id": 3230575,
                      "first_name": "Gary",
                      "last_name": "Brookes",
                      "email": "gary@example.com"
                    }
                  ]
                }
              }
            }
          },
          "default": {
            "description": "error payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "userId",
            "required": true,
            "in": "path",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "json payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "teammate_ids": {
                    "type": "array",
                    "description": "Full list of user IDs to be assigned to the Manager.",
                    "nullable": true,
                    "items": {
                      "type": "string"
                    }
                  }
                },
                "required": [
                  "teammate_ids"
                ]
              }
            }
          }
        }
      }
    }
  }
} as const
            