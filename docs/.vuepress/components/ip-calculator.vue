<template>
    <div class=container>
        <label for="serial">Serial Number:</label>
        <input id=serial v-model="serial" placeholder="Serial Number" />
        <!-- <input id=model-go type="radio" v-model="model" value="go"/><label for=model-go>Moku:Go</label>
        <input id=model-go type="radio" v-model="model" value="pro"/><label for=model-go>Moku:Pro</label> -->
        <p>
            IP Address:  <span class=addr>{{ ipAddress }}</span>
        </p>
        <p>
            MAC Address:  <span class=addr>{{ macStr }}</span>
        </p>
    </div>
</template>

<script>
export default {
    name: "IpCalculator",

    data () {
        return {
            serial: 0,
            model: 'go'
        }
    },

    computed: {
        ipAddress () {
            let ipv6 = new Array(4);
            let mac = this.binSplit(this.macAddress, 8, 6)

            ipv6[0] = (mac[0] ^ (1 << 1)) << 8 | mac[1]
            ipv6[1] =              mac[2] << 8 | 0xff
            ipv6[2] =                0xfe << 8 | mac[3]
            ipv6[3] =              mac[4] << 8 | mac[5]
            return 'fe80::' + ipv6.map( (n) => this.fixedWidthHex(n, 4)).join(':')
        },

        macAddress () {
            // In truth this is more than the base address, there's something a bit queer going on
            // and given that Pro doesn't do USB yet, we'll leave this as Go-only for now
            const mac_base = this.model == 'go' ? 0x706979B90000 : 0x706979B00000
            if (!parseInt(this.serial))
                return mac_base

            return mac_base + (this.serial * 4)
        },

        macStr () {
            return this.insertEvery(Number(this.macAddress).toString(16), 2, ':')
        }
    },

    methods: {
        fixedWidthHex (num, width) {
            let result = Number(num).toString(16)
            let padding = '0'.repeat(width - result.length)
            return padding + result
        },

        insertEvery (str, period, char) {
            let result = ''
            for (let i = 0; i < str.length; i += period) {
                result += str.slice(i, i + period)
                result += char
            }
            return result.slice(0, result.length - 1)
        },

        binSplit (num, bits, bins) {
            let res = Array()
            let mask = (1 << bits) - 1

            if (!bins) {
                bins = Math.ceil(Math.log2(num))
                bins = Math.ceil(bins / bits)
            }

            for (let i = 0; i < bins; i++) {
                res.push(num & mask)
                num = num / Math.pow(2, bits)
            }
            return res.reverse()
        }
    }
}
</script>

<style lang="stylus">
.container
    margin 1em
    padding 1em
    border 0.1em solid $borderColor

    .addr
        font-weight bold
</style>
